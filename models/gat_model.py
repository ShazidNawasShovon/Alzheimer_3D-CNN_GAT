import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.utils import softmax

class GATModel(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=64,
                 num_layers=3, num_heads=4, dropout=0.3):
        super(GATModel, self).__init__()
        # GAT layers
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(
            GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        # Last layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.dropout = dropout
        # Store attention weights for visualization
        self.attention_weights = {}
    def forward(self, data, return_attention_weights=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Clear previous attention weights
        if return_attention_weights:
            self.attention_weights = {}
        # Apply GAT layers
        for i, gat in enumerate(self.gat_layers):
            # Register hook to capture attention weights
            if return_attention_weights:
                attention_hook = self._create_attention_hook(i)
                handle = gat.register_forward_hook(attention_hook)
            # Forward pass
            x = gat(x, edge_index)
            # Remove hook after forward pass
            if return_attention_weights:
                handle.remove()
            if i < len(self.gat_layers) - 1:  # No activation for last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling
        x = global_mean_pool(x, batch)
        # Classification
        out = self.classifier(x)
        if return_attention_weights:
            return out, self.attention_weights
        else:
            return out
    def _create_attention_hook(self, layer_idx):
        """Create a hook to capture attention weights from GAT layer."""
        def hook(module, input, output):
            # GATConv outputs (output, attention_weights) when return_attention_weights=True
            if isinstance(output, tuple) and len(output) == 2:
                self.attention_weights[layer_idx] = output[1]
        return hook

class GATModelWithAttention(nn.Module):
    """GAT model that always returns attention weights."""
    def __init__(self, num_node_features, num_classes, hidden_dim=64,
                 num_layers=3, num_heads=4, dropout=0.3):
        super(GATModelWithAttention, self).__init__()
        # GAT layers
        self.gat_layers = nn.ModuleList()
        # First layer
        self.gat_layers.append(
            GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        # Last layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.dropout = dropout
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # Store attention weights
        attention_weights = []
        # Apply GAT layers
        for i, gat in enumerate(self.gat_layers):
            # Forward pass with attention weights
            x, att = gat(x, edge_index, return_attention_weights=True)
            # Extract attention values (second element of the tuple)
            attention_values = att[1]  # This is the actual attention tensor
            attention_weights.append(attention_values)
            if i < len(self.gat_layers) - 1:  # No activation for last layer
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Global pooling
        x = global_mean_pool(x, batch)
        # Classification
        out = self.classifier(x)
        return out, attention_weights

if __name__ == "__main__":
    from config import *
    from utils.gpu_utils import get_device
    # Test model
    print("Testing GAT model...")
    # Get device
    device = get_device()
    # Create a dummy input
    num_nodes = 100
    num_node_features = 3
    num_classes = 4
    x = torch.randn(num_nodes, num_node_features).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 200)).to(device)
    batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
    # Create model
    model = GATModel(num_node_features, num_classes).to(device)
    # Test forward pass
    output = model(Data(x=x, edge_index=edge_index, batch=batch))
    print(f"Model output shape: {output.shape}")
    # Test model with attention
    model_with_attention = GATModelWithAttention(num_node_features, num_classes).to(device)
    output, attention = model_with_attention(Data(x=x, edge_index=edge_index, batch=batch))
    print(f"Model with attention output shape: {output.shape}")
    print(f"Attention weights length: {len(attention)}")
    if len(attention) > 0:
        print(f"First attention weights type: {type(attention[0])}")
        print(f"First attention weights shape: {attention[0].shape}")
        print(f"First attention values: {attention[0][:5]}")
    else:
        print("No attention weights captured")
    print("GAT model test completed successfully!")