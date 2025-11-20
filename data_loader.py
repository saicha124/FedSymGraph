import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split


def generate_synthetic_network_graphs(num_graphs=100, num_nodes_per_graph=20, num_features=12):
    """
    Generate synthetic network traffic graphs for demonstration.
    In production, this would parse PCAP files or CSV network logs.
    
    Each graph represents a network flow snapshot:
    - Nodes: Network entities (hosts, services)
    - Edges: Communication patterns
    - Features: Traffic statistics (bytes, packets, flags, etc.)
    """
    graphs = []
    
    for i in range(num_graphs):
        num_nodes = np.random.randint(10, num_nodes_per_graph + 1)
        
        x = torch.randn(num_nodes, num_features)
        
        edge_prob = 0.3
        edge_list = []
        for src in range(num_nodes):
            for dst in range(num_nodes):
                if src != dst and np.random.rand() < edge_prob:
                    edge_list.append([src, dst])
        
        if len(edge_list) == 0:
            edge_list = [[0, 1], [1, 0]]
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        y = torch.tensor([1 if i % 3 == 0 else 0], dtype=torch.long)
        
        graph = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(graph)
    
    return graphs


def load_local_graphs(client_id, batch_size=8):
    """
    Load training and test data for a specific federated client.
    Each client has its own local dataset (simulating data heterogeneity).
    """
    np.random.seed(client_id * 42)
    torch.manual_seed(client_id * 42)
    
    all_graphs = generate_synthetic_network_graphs(num_graphs=100)
    
    train_graphs, test_graphs = train_test_split(all_graphs, test_size=0.2, random_state=client_id)
    
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def create_graph_from_network_flow(flow_dict):
    """
    Convert a network flow dictionary to a PyG Data object.
    Example flow_dict:
    {
        'src_ip': '192.168.1.10',
        'dst_ip': '10.0.0.5',
        'protocol': 'TCP',
        'src_port': 54321,
        'dst_port': 443,
        'bytes': 1500,
        'packets': 10,
        'flags': 'SYN,ACK'
    }
    """
    pass
