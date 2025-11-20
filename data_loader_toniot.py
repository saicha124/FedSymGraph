"""
TON_IoT Dataset Loader for FedSymGraph
Converts TON_IoT network flow CSV data into graph structures for GNN training.

Dataset: https://research.unsw.edu.au/projects/toniot-datasets
Kaggle: https://www.kaggle.com/datasets/amaniabourida/ton-iot
"""

import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class TONIoTGraphLoader:
    """
    Converts TON_IoT network flow data into graph structures.
    
    Graph Construction:
    - Nodes: Unique IP addresses (source and destination)
    - Edges: Network flows between IPs
    - Node Features: Aggregated traffic statistics per IP
    - Labels: Attack vs Normal classification
    """
    
    def __init__(self, csv_path, sample_size=None, attack_mapping=None):
        """
        Args:
            csv_path: Path to TON_IoT CSV file
            sample_size: Number of flows to sample (None = use all)
            attack_mapping: Dict to map attack types to MITRE tactics
        """
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        
        # MITRE ATT&CK mapping for TON_IoT attack types
        self.attack_mapping = attack_mapping or {
            'ddos': 'T1498: Network Denial of Service',
            'dos': 'T1499: Endpoint Denial of Service',
            'password': 'T1110: Brute Force',
            'scanning': 'T1046: Network Service Scanning',
            'xss': 'T1189: Drive-by Compromise',
            'injection': 'T1190: Exploit Public-Facing Application',
            'backdoor': 'T1546: Persistence via Backdoor',
            'ransomware': 'T1486: Data Encrypted for Impact',
            'mitm': 'T1557: Man-in-the-Middle'
        }
        
        # Network flow features to use (42 total in TON_IoT)
        self.flow_features = [
            'proto', 'duration', 'bytes', 'packets',
            'src_bytes', 'dst_bytes', 'src_pkts', 'dst_pkts'
        ]
    
    def load_csv(self):
        """Load TON_IoT CSV data."""
        print(f"Loading TON_IoT dataset from {self.csv_path}...")
        
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"TON_IoT dataset not found at {self.csv_path}\n"
                f"Download from: https://www.kaggle.com/datasets/amaniabourida/ton-iot"
            )
        
        df = pd.read_csv(self.csv_path, nrows=self.sample_size)
        print(f"Loaded {len(df)} network flows")
        print(f"Attack distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def preprocess(self, df):
        """Clean and preprocess TON_IoT data."""
        # Handle missing values
        df = df.fillna(0)
        
        # Ensure required columns exist
        required_cols = ['src_ip', 'dst_ip', 'label']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            # Try common alternative column names
            rename_map = {
                'srcip': 'src_ip', 'dstip': 'dst_ip',
                'src': 'src_ip', 'dst': 'dst_ip',
                'type': 'attack_type'
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Binary label: 0=normal, 1=attack
        if 'label' in df.columns:
            df['binary_label'] = (df['label'] != 0).astype(int)
        else:
            df['binary_label'] = 0
        
        return df
    
    def create_graph_from_flows(self, flows_df, window_size=100):
        """
        Convert a set of network flows into a graph.
        
        Args:
            flows_df: DataFrame of network flows
            window_size: Number of flows per graph
        
        Returns:
            PyTorch Geometric Data object
        """
        # Create IP to node ID mapping
        unique_ips = list(set(flows_df['src_ip'].unique()) | set(flows_df['dst_ip'].unique()))
        ip_to_id = {ip: idx for idx, ip in enumerate(unique_ips)}
        num_nodes = len(unique_ips)
        
        # Build edge list and edge features
        edge_list = []
        edge_attrs = []
        
        for _, flow in flows_df.iterrows():
            src_id = ip_to_id.get(flow['src_ip'], 0)
            dst_id = ip_to_id.get(flow['dst_ip'], 0)
            
            edge_list.append([src_id, dst_id])
            
            # Extract flow features
            flow_feats = []
            for feat in self.flow_features:
                if feat in flow:
                    flow_feats.append(float(flow[feat]))
                else:
                    flow_feats.append(0.0)
            edge_attrs.append(flow_feats)
        
        if len(edge_list) == 0:
            # Create self-loop if no edges
            edge_list = [[0, 0]]
            edge_attrs = [[0.0] * len(self.flow_features)]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # Create node features (aggregate incoming/outgoing traffic)
        node_features = np.zeros((num_nodes, 12))
        
        for idx, ip in enumerate(unique_ips):
            # Outgoing traffic
            out_flows = flows_df[flows_df['src_ip'] == ip]
            if len(out_flows) > 0:
                node_features[idx, 0] = len(out_flows)  # num outgoing connections
                node_features[idx, 1] = out_flows['bytes'].sum() if 'bytes' in out_flows else 0
                node_features[idx, 2] = out_flows['packets'].sum() if 'packets' in out_flows else 0
                node_features[idx, 3] = out_flows['duration'].mean() if 'duration' in out_flows else 0
            
            # Incoming traffic
            in_flows = flows_df[flows_df['dst_ip'] == ip]
            if len(in_flows) > 0:
                node_features[idx, 4] = len(in_flows)  # num incoming connections
                node_features[idx, 5] = in_flows['bytes'].sum() if 'bytes' in in_flows else 0
                node_features[idx, 6] = in_flows['packets'].sum() if 'packets' in in_flows else 0
                node_features[idx, 7] = in_flows['duration'].mean() if 'duration' in in_flows else 0
            
            # Protocol distribution (simplified)
            node_features[idx, 8] = (flows_df['src_ip'] == ip).sum()  # total flows from this IP
            node_features[idx, 9] = (flows_df['dst_ip'] == ip).sum()  # total flows to this IP
            
            # Attack indicators
            ip_flows = flows_df[(flows_df['src_ip'] == ip) | (flows_df['dst_ip'] == ip)]
            node_features[idx, 10] = ip_flows['binary_label'].sum()  # attack count
            node_features[idx, 11] = ip_flows['binary_label'].mean()  # attack ratio
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Graph-level label (1 if any flow is attack, 0 otherwise)
        y = torch.tensor([flows_df['binary_label'].max()], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def load_graphs(self, flows_per_graph=100, max_graphs=1000):
        """
        Load TON_IoT data and convert to graphs.
        
        Args:
            flows_per_graph: Number of flows to include per graph
            max_graphs: Maximum number of graphs to create
        
        Returns:
            List of PyTorch Geometric Data objects
        """
        df = self.load_csv()
        df = self.preprocess(df)
        
        # Normalize numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['binary_label', 'label']]
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols].fillna(0))
        
        graphs = []
        num_flows = len(df)
        
        # Create graphs by sliding window
        for i in range(0, min(num_flows, max_graphs * flows_per_graph), flows_per_graph):
            window_df = df.iloc[i:i+flows_per_graph]
            
            if len(window_df) < 10:  # Skip very small windows
                continue
            
            try:
                graph = self.create_graph_from_flows(window_df, flows_per_graph)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to create graph from window {i}: {e}")
                continue
            
            if len(graphs) >= max_graphs:
                break
        
        print(f"Created {len(graphs)} graphs from TON_IoT data")
        attack_graphs = sum(1 for g in graphs if g.y.item() == 1)
        print(f"Attack graphs: {attack_graphs}, Normal graphs: {len(graphs) - attack_graphs}")
        
        return graphs


def load_toniot_for_client(client_id, csv_path, batch_size=8, flows_per_graph=100):
    """
    Load TON_IoT data for a specific federated client.
    Each client gets a different subset of the data.
    
    Args:
        client_id: Unique client identifier
        csv_path: Path to TON_IoT CSV file
        batch_size: Batch size for DataLoader
        flows_per_graph: Flows per graph
    
    Returns:
        train_loader, test_loader
    """
    # Set seed for reproducibility per client
    np.random.seed(client_id * 42)
    torch.manual_seed(client_id * 42)
    
    loader = TONIoTGraphLoader(csv_path, sample_size=10000 * client_id)
    
    all_graphs = loader.load_graphs(flows_per_graph=flows_per_graph, max_graphs=200)
    
    if len(all_graphs) == 0:
        raise ValueError(f"No graphs created for client {client_id}. Check dataset path.")
    
    # Split into train/test
    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=client_id)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    # Test with sample data
    csv_path = "TON_IoT_Network.csv"  # Update with actual path
    
    if os.path.exists(csv_path):
        print("Testing TON_IoT loader...")
        train_loader, test_loader = load_toniot_for_client(
            client_id=1,
            csv_path=csv_path,
            batch_size=4,
            flows_per_graph=50
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Inspect first batch
        for batch in train_loader:
            print(f"\nBatch info:")
            print(f"  Nodes: {batch.x.shape}")
            print(f"  Edges: {batch.edge_index.shape}")
            print(f"  Labels: {batch.y}")
            break
    else:
        print(f"Dataset not found at {csv_path}")
        print("Download from: https://www.kaggle.com/datasets/amaniabourida/ton-iot")
