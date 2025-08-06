import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax


class ImprovedGATModel(nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_dim=128,
                 num_layers=4, num_heads=8, dropout=0.3):
        super(ImprovedGATModel, self).__init__()

        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        self.norm_layers.append(nn.LayerNorm(hidden_dim * num_heads))
        self.dropout_layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim * num_heads))
            self.dropout_layers.append(nn.Dropout(dropout))

        # Last layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))

        # Enhanced classifier with more layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because we concatenate mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Skip connection projection
        self.skip_proj = nn.Linear(num_node_features, hidden_dim * num_heads)

        self.dropout = dropout
        self.num_layers = num_layers

        # Store attention weights for visualization
        self.attention_weights = {}

    def forward(self, data, return_attention_weights=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Clear previous attention weights
        if return_attention_weights:
            self.attention_weights = {}

        # Initial skip connection
        skip = self.skip_proj(x)

        # Apply GAT layers with residual connections
        for i, (gat, norm, dropout_layer) in enumerate(
                zip(self.gat_layers, self.norm_layers, self.dropout_layers)
        ):
            # Register hook to capture attention weights
            if return_attention_weights:
                attention_hook = self._create_attention_hook(i)
                handle = gat.register_forward_hook(attention_hook)

            # Forward pass
            x_new = gat(x, edge_index)

            # Remove hook after forward pass
            if return_attention_weights:
                handle.remove()

            # Apply normalization and activation
            x_new = norm(x_new)
            if i < self.num_layers - 1:  # No activation for last layer
                x_new = F.elu(x_new)
                x_new = dropout_layer(x_new)

            # Add residual connection
            if i == 0:
                x = x_new + skip
            else:
                x = x_new + x

            # Apply dropout after residual
            x = dropout_layer(x)

        # Global pooling - combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

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


class ImprovedGATModelWithAttention(nn.Module):
    """Improved GAT model that always returns attention weights."""

    def __init__(self, num_node_features, num_classes, hidden_dim=128,
                 num_layers=4, num_heads=8, dropout=0.3):
        super(ImprovedGATModelWithAttention, self).__init__()

        # GAT layers with residual connections
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # First layer
        self.gat_layers.append(
            GATConv(num_node_features, hidden_dim, heads=num_heads, dropout=dropout)
        )
        self.norm_layers.append(nn.LayerNorm(hidden_dim * num_heads))
        self.dropout_layers.append(nn.Dropout(dropout))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim * num_heads))
            self.dropout_layers.append(nn.Dropout(dropout))

        # Last layer
        self.gat_layers.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False, dropout=dropout)
        )
        self.norm_layers.append(nn.LayerNorm(hidden_dim))
        self.dropout_layers.append(nn.Dropout(dropout))

        # Enhanced classifier with more layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because we concatenate mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Skip connection projection
        self.skip_proj = nn.Linear(num_node_features, hidden_dim * num_heads)

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Store attention weights
        attention_weights = []

        # Initial skip connection
        skip = self.skip_proj(x)

        # Apply GAT layers with residual connections
        for i, (gat, norm, dropout_layer) in enumerate(
                zip(self.gat_layers, self.norm_layers, self.dropout_layers)
        ):
            # Forward pass with attention weights
            x_new, att = gat(x, edge_index, return_attention_weights=True)

            # Extract attention values
            attention_values = att[1]
            attention_weights.append(attention_values)

            # Apply normalization and activation
            x_new = norm(x_new)
            if i < self.num_layers - 1:  # No activation for last layer
                x_new = F.elu(x_new)
                x_new = dropout_layer(x_new)

            # Add residual connection
            if i == 0:
                x = x_new + skip
            else:
                x = x_new + x

            # Apply dropout after residual
            x = dropout_layer(x)

        # Global pooling - combine mean and max pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Classification
        out = self.classifier(x)

        return out, attention_weights


if __name__ == "__main__":
    from config import *
    from utils.gpu_utils import get_device

    # Test model
    print("Testing Improved GAT model...")

    # Get device
    device = get_device()

    # Create a dummy input
    num_nodes = 100
    num_node_features = 64  # Increased to match CNN features
    num_classes = 4
    x = torch.randn(num_nodes, num_node_features).to(device)
    edge_index = torch.randint(0, num_nodes, (2, 200)).to(device)
    batch = torch.zeros(num_nodes, dtype=torch.long).to(device)

    # Create model
    model = ImprovedGATModel(num_node_features, num_classes).to(device)

    # Test forward pass
    output = model(Data(x=x, edge_index=edge_index, batch=batch))
    print(f"Model output shape: {output.shape}")

    # Test model with attention
    model_with_attention = ImprovedGATModelWithAttention(num_node_features, num_classes).to(device)
    output, attention = model_with_attention(Data(x=x, edge_index=edge_index, batch=batch))
    print(f"Model with attention output shape: {output.shape}")
    print(f"Attention weights length: {len(attention)}")

    if len(attention) > 0:
        print(f"First attention weights shape: {attention[0].shape}")
    else:
        print("No attention weights captured")

    print("Improved GAT model test completed successfully!")