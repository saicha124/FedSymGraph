"""
Heterogeneous Graph Loader for FedSymGraph Paper Approach
Creates dynamic heterogeneous graphs with multiple node types (IPs, ports, protocols)
for intrusion detection using TON-IoT dataset.
"""

import os
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class HeterogeneousGraphLoader:
    """
    Creates heterogeneous graphs from TON_IoT network traffic.
    
    Graph Structure:
    - Node Types: 
        * 'ip' - IP addresses (hosts/servers)
        * 'port' - Port numbers (services)
        * 'protocol' - Protocol types (TCP, UDP, ICMP)
    - Edge Types:
        * ('ip', 'connects_to', 'ip') - Direct IP communication
        * ('ip', 'uses', 'port') - Host uses port
        * ('port', 'runs', 'protocol') - Port runs on protocol
        * ('ip', 'sends', 'ip') - Temporal flow direction
    """
    
    def __init__(self, csv_path, sample_size=None):
        self.csv_path = csv_path
        self.sample_size = sample_size
        self.scaler = StandardScaler()
        
        # Protocol encoding
        self.protocol_map = {
            'TCP': 0, 'tcp': 0,
            'UDP': 1, 'udp': 1,
            'ICMP': 2, 'icmp': 2,
            'HTTP': 3, 'http': 3,
            'HTTPS': 4, 'https': 4,
            'DNS': 5, 'dns': 5,
            'SSH': 6, 'ssh': 6
        }
        
        # MITRE ATT&CK mapping
        self.attack_mapping = {
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
    
    def load_csv(self):
        """Load TON_IoT dataset."""
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
        """Preprocess TON_IoT data - preserve raw port numbers for categorical features."""
        df = df.fillna(0)
        
        # Standardize column names
        rename_map = {
            'srcip': 'src_ip', 'dstip': 'dst_ip',
            'src': 'src_ip', 'dst': 'dst_ip',
            'sport': 'src_port', 'dsport': 'dst_port',
            'type': 'attack_type'
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        
        # Binary label
        if 'label' in df.columns:
            df['binary_label'] = (df['label'] != 0).astype(int)
        else:
            df['binary_label'] = 0
        
        # Ensure port columns exist (keep as integers)
        if 'src_port' not in df.columns:
            df['src_port'] = np.random.randint(1024, 65535, size=len(df))
        if 'dst_port' not in df.columns:
            df['dst_port'] = np.random.randint(1, 1024, size=len(df))
        
        # Convert ports to integers (important: before scaling!)
        df['src_port'] = df['src_port'].astype(int)
        df['dst_port'] = df['dst_port'].astype(int)
        
        # Encode protocols
        if 'proto' in df.columns:
            df['proto_encoded'] = df['proto'].map(self.protocol_map).fillna(0).astype(int)
        else:
            df['proto_encoded'] = 0
        
        return df
    
    def create_heterogeneous_graph(self, flows_df, window_size=100):
        """
        Create a heterogeneous graph from network flows.
        
        Args:
            flows_df: DataFrame of network flows
            window_size: Number of flows per graph
        
        Returns:
            HeteroData object with multiple node and edge types
        """
        data = HeteroData()
        
        # === Create Node Mappings ===
        
        # IP nodes
        unique_ips = list(set(flows_df['src_ip'].unique()) | set(flows_df['dst_ip'].unique()))
        ip_to_id = {ip: idx for idx, ip in enumerate(unique_ips)}
        num_ip_nodes = len(unique_ips)
        
        # Port nodes
        unique_ports = list(set(flows_df['src_port'].unique()) | set(flows_df['dst_port'].unique()))
        port_to_id = {port: idx for idx, port in enumerate(unique_ports)}
        num_port_nodes = len(unique_ports)
        
        # Protocol nodes
        unique_protocols = flows_df['proto_encoded'].unique()
        protocol_to_id = {proto: idx for idx, proto in enumerate(unique_protocols)}
        num_protocol_nodes = len(unique_protocols)
        
        # === Build Node Features ===
        
        # IP node features (12 features)
        ip_features = np.zeros((num_ip_nodes, 12))
        for idx, ip in enumerate(unique_ips):
            out_flows = flows_df[flows_df['src_ip'] == ip]
            in_flows = flows_df[flows_df['dst_ip'] == ip]
            ip_flows = flows_df[(flows_df['src_ip'] == ip) | (flows_df['dst_ip'] == ip)]
            
            if len(out_flows) > 0:
                ip_features[idx, 0] = len(out_flows)  # outgoing connections
                ip_features[idx, 1] = out_flows['bytes'].sum() if 'bytes' in out_flows else 0
                ip_features[idx, 2] = out_flows['packets'].sum() if 'packets' in out_flows else 0
            
            if len(in_flows) > 0:
                ip_features[idx, 3] = len(in_flows)  # incoming connections
                ip_features[idx, 4] = in_flows['bytes'].sum() if 'bytes' in in_flows else 0
                ip_features[idx, 5] = in_flows['packets'].sum() if 'packets' in in_flows else 0
            
            ip_features[idx, 6] = len(out_flows.get('src_port', []).unique()) if len(out_flows) > 0 else 0  # unique ports used
            ip_features[idx, 7] = len(in_flows.get('dst_port', []).unique()) if len(in_flows) > 0 else 0
            ip_features[idx, 8] = ip_flows['duration'].mean() if 'duration' in ip_flows and len(ip_flows) > 0 else 0
            ip_features[idx, 9] = ip_flows['proto_encoded'].nunique()  # protocol diversity
            ip_features[idx, 10] = ip_flows['binary_label'].sum()  # attack count
            ip_features[idx, 11] = ip_flows['binary_label'].mean() if len(ip_flows) > 0 else 0  # attack ratio
        
        data['ip'].x = torch.tensor(ip_features, dtype=torch.float)
        
        # Port node features (6 features)
        port_features = np.zeros((num_port_nodes, 6))
        for idx, port in enumerate(unique_ports):
            port_flows = flows_df[(flows_df['src_port'] == port) | (flows_df['dst_port'] == port)]
            
            port_features[idx, 0] = port  # port number
            port_features[idx, 1] = len(port_flows)  # flow count
            port_features[idx, 2] = port_flows['bytes'].sum() if 'bytes' in port_flows else 0
            port_features[idx, 3] = port_flows['packets'].sum() if 'packets' in port_flows else 0
            port_features[idx, 4] = 1 if port < 1024 else 0  # is well-known port
            port_features[idx, 5] = port_flows['binary_label'].mean() if len(port_flows) > 0 else 0
        
        data['port'].x = torch.tensor(port_features, dtype=torch.float)
        
        # Protocol node features (5 features)
        protocol_features = np.zeros((num_protocol_nodes, 5))
        for idx, proto in enumerate(unique_protocols):
            proto_flows = flows_df[flows_df['proto_encoded'] == proto]
            
            protocol_features[idx, 0] = proto  # protocol ID
            protocol_features[idx, 1] = len(proto_flows)  # flow count
            protocol_features[idx, 2] = proto_flows['bytes'].sum() if 'bytes' in proto_flows else 0
            protocol_features[idx, 3] = proto_flows['packets'].sum() if 'packets' in proto_flows else 0
            protocol_features[idx, 4] = proto_flows['binary_label'].mean() if len(proto_flows) > 0 else 0
        
        data['protocol'].x = torch.tensor(protocol_features, dtype=torch.float)
        
        # === Build Edges ===
        
        # (ip, connects_to, ip) - bidirectional communication
        ip_to_ip_edges = []
        for _, flow in flows_df.iterrows():
            src_id = ip_to_id[flow['src_ip']]
            dst_id = ip_to_id[flow['dst_ip']]
            ip_to_ip_edges.append([src_id, dst_id])
        
        if len(ip_to_ip_edges) > 0:
            data['ip', 'connects_to', 'ip'].edge_index = torch.tensor(ip_to_ip_edges, dtype=torch.long).t().contiguous()
        
        # (ip, uses, port) - IP uses port
        ip_to_port_edges = []
        for _, flow in flows_df.iterrows():
            src_id = ip_to_id[flow['src_ip']]
            port_id = port_to_id[flow['src_port']]
            ip_to_port_edges.append([src_id, port_id])
        
        if len(ip_to_port_edges) > 0:
            data['ip', 'uses', 'port'].edge_index = torch.tensor(ip_to_port_edges, dtype=torch.long).t().contiguous()
        
        # (port, runs, protocol) - Port runs on protocol
        port_to_protocol_edges = []
        for _, flow in flows_df.iterrows():
            port_id = port_to_id[flow['dst_port']]
            proto_id = protocol_to_id[flow['proto_encoded']]
            port_to_protocol_edges.append([port_id, proto_id])
        
        if len(port_to_protocol_edges) > 0:
            data['port', 'runs', 'protocol'].edge_index = torch.tensor(port_to_protocol_edges, dtype=torch.long).t().contiguous()
        
        # === Add REVERSE Edges for bidirectional message passing ===
        
        # (port, used_by, ip) - Reverse of (ip, uses, port)
        if len(ip_to_port_edges) > 0:
            port_to_ip_edges = [[edge[1], edge[0]] for edge in ip_to_port_edges]
            data['port', 'used_by', 'ip'].edge_index = torch.tensor(port_to_ip_edges, dtype=torch.long).t().contiguous()
        
        # (protocol, served_by, port) - Reverse of (port, runs, protocol)
        if len(port_to_protocol_edges) > 0:
            protocol_to_port_edges = [[edge[1], edge[0]] for edge in port_to_protocol_edges]
            data['protocol', 'served_by', 'port'].edge_index = torch.tensor(protocol_to_port_edges, dtype=torch.long).t().contiguous()
        
        # Graph-level label
        data.y = torch.tensor([flows_df['binary_label'].max()], dtype=torch.long)
        
        # Store metadata for reasoning
        data.metadata_dict = {
            'num_flows': len(flows_df),
            'attack_types': flows_df[flows_df['binary_label'] == 1]['attack_type'].unique().tolist() if 'attack_type' in flows_df.columns else [],
            'num_unique_ips': num_ip_nodes,
            'num_unique_ports': num_port_nodes,
            'num_protocols': num_protocol_nodes
        }
        
        return data
    
    def load_graphs(self, flows_per_graph=100, max_graphs=500):
        """
        Load TON_IoT data and create heterogeneous graphs.
        
        Args:
            flows_per_graph: Number of flows per graph
            max_graphs: Maximum number of graphs to create
        
        Returns:
            List of HeteroData objects
        """
        df = self.load_csv()
        df = self.preprocess(df)
        
        # Normalize numerical features (EXCLUDE port numbers to preserve categorical meaning)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['binary_label', 'label', 'proto_encoded', 'src_port', 'dst_port']]
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols].fillna(0))
        
        graphs = []
        num_flows = len(df)
        
        # Create graphs by sliding window
        for i in range(0, min(num_flows, max_graphs * flows_per_graph), flows_per_graph):
            window_df = df.iloc[i:i+flows_per_graph]
            
            if len(window_df) < 10:
                continue
            
            try:
                graph = self.create_heterogeneous_graph(window_df, flows_per_graph)
                graphs.append(graph)
            except Exception as e:
                print(f"Warning: Failed to create graph from window {i}: {e}")
                continue
            
            if len(graphs) >= max_graphs:
                break
        
        print(f"Created {len(graphs)} heterogeneous graphs from TON_IoT data")
        attack_graphs = sum(1 for g in graphs if g.y.item() == 1)
        print(f"Attack graphs: {attack_graphs}, Normal graphs: {len(graphs) - attack_graphs}")
        
        return graphs


def load_heterogeneous_for_client(client_id, csv_path, batch_size=8, flows_per_graph=100):
    """
    Load heterogeneous graphs for a federated client.
    
    Args:
        client_id: Client identifier
        csv_path: Path to TON_IoT CSV
        batch_size: Batch size
        flows_per_graph: Flows per graph
    
    Returns:
        train_loader, test_loader
    """
    np.random.seed(client_id * 42)
    torch.manual_seed(client_id * 42)
    
    loader = HeterogeneousGraphLoader(csv_path, sample_size=10000 * client_id)
    all_graphs = loader.load_graphs(flows_per_graph=flows_per_graph, max_graphs=200)
    
    if len(all_graphs) == 0:
        raise ValueError(f"No graphs created for client {client_id}. Check dataset.")
    
    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=client_id)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# Test script
if __name__ == "__main__":
    csv_path = "TON_IoT_Network.csv"
    
    if os.path.exists(csv_path):
        print("Testing Heterogeneous Graph Loader...")
        train_loader, test_loader = load_heterogeneous_for_client(
            client_id=1,
            csv_path=csv_path,
            batch_size=4,
            flows_per_graph=50
        )
        
        print(f"\nTrain batches: {len(train_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Inspect first batch
        for batch in train_loader:
            print(f"\nHeterogeneous Graph Structure:")
            print(f"  Node types: {batch.node_types}")
            print(f"  Edge types: {batch.edge_types}")
            print(f"  IP nodes: {batch['ip'].x.shape}")
            print(f"  Port nodes: {batch['port'].x.shape}")
            print(f"  Protocol nodes: {batch['protocol'].x.shape}")
            print(f"  Labels: {batch.y}")
            break
    else:
        print(f"Dataset not found at {csv_path}")
