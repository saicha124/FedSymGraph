"""
Heterogeneous Graph Neural Network for FedSymGraph Paper Approach
Handles multiple node types (IPs, ports, protocols) and edge types.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, Linear
from torch_geometric.nn import global_mean_pool


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous Graph Neural Network for intrusion detection.
    
    Processes multiple node types:
    - IP nodes (hosts/servers)
    - Port nodes (services)
    - Protocol nodes (communication protocols)
    
    Uses different message passing for different edge types:
    - (ip, connects_to, ip) - IP communication
    - (ip, uses, port) - Host-service relation
    - (port, runs, protocol) - Service-protocol relation
    """
    
    def __init__(self, hidden_channels=64, num_classes=2, num_layers=2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # Input projections for each node type
        self.ip_lin = Linear(12, hidden_channels)  # IP features: 12
        self.port_lin = Linear(6, hidden_channels)  # Port features: 6
        self.protocol_lin = Linear(5, hidden_channels)  # Protocol features: 5
        
        # Heterogeneous convolution layers (with reverse edges for bidirectional message passing)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # Forward edges
                ('ip', 'connects_to', 'ip'): GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
                ('ip', 'uses', 'port'): GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
                ('port', 'runs', 'protocol'): GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
                # Reverse edges (critical for IP nodes to receive port/protocol information)
                ('port', 'used_by', 'ip'): GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
                ('protocol', 'served_by', 'port'): GATv2Conv(hidden_channels, hidden_channels, heads=4, concat=False, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Classification head (operates on IP nodes for graph-level prediction)
        self.classifier = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden_channels // 2, num_classes)
        )
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """
        Forward pass through heterogeneous graph.
        
        Args:
            x_dict: Dictionary of node features {node_type: tensor}
            edge_index_dict: Dictionary of edge indices {edge_type: tensor}
            batch_dict: Dictionary of batch assignments {node_type: tensor}
        
        Returns:
            Graph-level predictions
        """
        # Project input features to common dimension
        x_dict = {
            'ip': self.ip_lin(x_dict['ip']),
            'port': self.port_lin(x_dict['port']),
            'protocol': self.protocol_lin(x_dict['protocol'])
        }
        
        # Apply heterogeneous convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        # Graph-level pooling on IP nodes (primary entity type)
        if batch_dict is not None and 'ip' in batch_dict:
            # Batch mode: pool over IP nodes per graph
            ip_embedding = global_mean_pool(x_dict['ip'], batch_dict['ip'])
        else:
            # Single graph: mean of all IP nodes
            ip_embedding = x_dict['ip'].mean(dim=0, keepdim=True)
        
        # Classification
        out = self.classifier(ip_embedding)
        
        return out
    
    def get_embeddings(self, x_dict, edge_index_dict):
        """
        Extract node embeddings for explainability.
        
        Returns:
            Dictionary of learned node embeddings
        """
        # Project input features
        x_dict = {
            'ip': self.ip_lin(x_dict['ip']),
            'port': self.port_lin(x_dict['port']),
            'protocol': self.protocol_lin(x_dict['protocol'])
        }
        
        # Apply convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        return x_dict


class HeteroGNNWithAttention(torch.nn.Module):
    """
    Enhanced Heterogeneous GNN with attention mechanism for explainability.
    Includes attention weights for interpretation.
    """
    
    def __init__(self, hidden_channels=64, num_classes=2, num_layers=2):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_classes = num_classes
        
        # Input projections
        self.ip_lin = Linear(12, hidden_channels)
        self.port_lin = Linear(6, hidden_channels)
        self.protocol_lin = Linear(5, hidden_channels)
        
        # Heterogeneous convolutions with attention (with reverse edges)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                # Forward edges
                ('ip', 'connects_to', 'ip'): GATv2Conv(
                    hidden_channels, hidden_channels, 
                    heads=4, concat=False, 
                    add_self_loops=False,
                    return_attention_weights=True
                ),
                ('ip', 'uses', 'port'): GATv2Conv(
                    hidden_channels, hidden_channels,
                    heads=4, concat=False,
                    add_self_loops=False
                ),
                ('port', 'runs', 'protocol'): GATv2Conv(
                    hidden_channels, hidden_channels,
                    heads=4, concat=False,
                    add_self_loops=False
                ),
                # Reverse edges (critical for bidirectional message passing)
                ('port', 'used_by', 'ip'): GATv2Conv(
                    hidden_channels, hidden_channels,
                    heads=4, concat=False,
                    add_self_loops=False
                ),
                ('protocol', 'served_by', 'port'): GATv2Conv(
                    hidden_channels, hidden_channels,
                    heads=4, concat=False,
                    add_self_loops=False
                ),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Attention-based pooling for graph representation
        self.attention_pool = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.Tanh(),
            Linear(hidden_channels // 2, 1)
        )
        
        # Classifier
        self.classifier = torch.nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            Linear(hidden_channels // 2, num_classes)
        )
        
        # Store attention weights for explainability
        self.attention_weights = None
    
    def forward(self, x_dict, edge_index_dict, batch_dict=None):
        """Forward pass with attention tracking."""
        # Project features
        x_dict = {
            'ip': self.ip_lin(x_dict['ip']),
            'port': self.port_lin(x_dict['port']),
            'protocol': self.protocol_lin(x_dict['protocol'])
        }
        
        # Apply convolutions
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        
        # Attention-based pooling on IP nodes
        ip_features = x_dict['ip']
        attention_scores = self.attention_pool(ip_features)
        attention_weights = F.softmax(attention_scores, dim=0)
        self.attention_weights = attention_weights  # Store for explainability
        
        if batch_dict is not None and 'ip' in batch_dict:
            # Weighted pooling per graph
            weighted_features = ip_features * attention_weights
            ip_embedding = global_mean_pool(weighted_features, batch_dict['ip'])
        else:
            # Single graph weighted average
            ip_embedding = (ip_features * attention_weights).sum(dim=0, keepdim=True)
        
        # Classification
        out = self.classifier(ip_embedding)
        
        return out


# Wrapper for backward compatibility with existing federated client
class FedHeteroGNN(torch.nn.Module):
    """
    Wrapper for heterogeneous GNN to work with federated learning client.
    Adapts HeteroData format to work with existing training loop.
    """
    
    def __init__(self, hidden_channels=64, num_classes=2, num_layers=2):
        super().__init__()
        self.model = HeteroGNN(hidden_channels, num_classes, num_layers)
    
    def forward(self, data):
        """
        Forward pass compatible with HeteroData batches.
        
        Args:
            data: HeteroData batch object
        
        Returns:
            Classification logits
        """
        # Extract node features, edge indices, and batch assignments
        x_dict = {
            'ip': data['ip'].x,
            'port': data['port'].x,
            'protocol': data['protocol'].x
        }
        
        edge_index_dict = {
            ('ip', 'connects_to', 'ip'): data['ip', 'connects_to', 'ip'].edge_index,
            ('ip', 'uses', 'port'): data['ip', 'uses', 'port'].edge_index,
            ('port', 'runs', 'protocol'): data['port', 'runs', 'protocol'].edge_index,
            ('port', 'used_by', 'ip'): data['port', 'used_by', 'ip'].edge_index,
            ('protocol', 'served_by', 'port'): data['protocol', 'served_by', 'port'].edge_index
        }
        
        batch_dict = {
            'ip': data['ip'].batch if hasattr(data['ip'], 'batch') else None,
            'port': data['port'].batch if hasattr(data['port'], 'batch') else None,
            'protocol': data['protocol'].batch if hasattr(data['protocol'], 'batch') else None
        }
        
        return self.model(x_dict, edge_index_dict, batch_dict)


# Test script
if __name__ == "__main__":
    print("Testing Heterogeneous GNN...")
    
    # Create dummy heterogeneous graph
    from torch_geometric.data import HeteroData
    
    data = HeteroData()
    
    # IP nodes
    data['ip'].x = torch.randn(10, 12)
    
    # Port nodes
    data['port'].x = torch.randn(5, 6)
    
    # Protocol nodes
    data['protocol'].x = torch.randn(3, 5)
    
    # Edges (forward and reverse)
    data['ip', 'connects_to', 'ip'].edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data['ip', 'uses', 'port'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data['port', 'runs', 'protocol'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    # Reverse edges
    data['port', 'used_by', 'ip'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    data['protocol', 'served_by', 'port'].edge_index = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    
    # Batch assignments
    data['ip'].batch = torch.zeros(10, dtype=torch.long)
    data['port'].batch = torch.zeros(5, dtype=torch.long)
    data['protocol'].batch = torch.zeros(3, dtype=torch.long)
    
    # Test model
    model = FedHeteroGNN(hidden_channels=64, num_classes=2, num_layers=2)
    model.eval()
    
    with torch.no_grad():
        out = model(data)
        print(f"Output shape: {out.shape}")
        print(f"Predictions: {out}")
        print(f"Predicted class: {out.argmax(dim=1)}")
    
    print("\n✓ Heterogeneous GNN model working correctly!")
