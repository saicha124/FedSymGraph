import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class FedGNN(torch.nn.Module):
    """
    Federated Graph Neural Network for intrusion detection.
    Uses Graph Attention Networks (GAT) to capture structural dependencies
    in network traffic patterns.
    """
    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads=2)
        self.conv2 = GATv2Conv(hidden_channels * 2, hidden_channels, heads=1)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        
        x = global_mean_pool(x, batch)
        
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin(x)
