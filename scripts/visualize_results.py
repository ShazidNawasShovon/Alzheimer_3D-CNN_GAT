import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.data_loader import MRIDataset
from models.gat_model import GATModel
from utils.visualization import plot_attention_weights


def visualize_attention(model, data_loader, device, num_samples=1):
    model.eval()

    # Get a batch of data
    data = next(iter(data_loader))
    data = data.to(device)

    # Hook to capture attention weights
    attention_weights = []

    def hook(module, input, output):
        attention_weights.append(output[1])  # Attention weights are the second output

    # Register hook on the last GAT layer
    handle = model.gat_layers[-1].attentions.register_forward_hook(hook)

    # Forward pass
    with torch.no_grad():
        _ = model(data)

    # Remove hook
    handle.remove()

    # Get attention weights (shape: [num_edges, num_heads])
    att_weights = attention_weights[0].cpu().numpy()

    # Average attention across heads
    avg_att = np.mean(att_weights, axis=1)

    # Get edge indices
    edge_index = data.edge_index.cpu().numpy()

    # Create a dictionary to store node attention scores
    node_attention = {}

    # Sum attention weights for each node
    for i, (src, dst) in enumerate(edge_index.T):
        if src not in node_attention:
            node_attention[src] = 0
        if dst not in node_attention:
            node_attention[dst] = 0

        node_attention[src] += avg_att[i]
        node_attention[dst] += avg_att[i]

    # Plot attention weights
    plot_attention_weights(node_attention, top_n=20, save_path=os.path.join(PLOT_SAVE_DIR, 'attention_weights.png'))

    # Print top 10 important nodes
    sorted_nodes = sorted(node_attention.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Important Brain Regions (by attention score):")
    for i, (node_idx, score) in enumerate(sorted_nodes[:10]):
        print(f"{i + 1}. ROI {node_idx}: {score:.4f}")


def main():
    # Load test dataset
    dataset = MRIDataset(PROCESSED_DATA_DIR, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Load model
    num_node_features = dataset[0].x.shape[1]
    model = GATModel(num_node_features, len(CLASSES),
                     hidden_dim=GAT_HIDDEN_DIM,
                     num_layers=NUM_GAT_LAYERS,
                     num_heads=NUM_HEADS,
                     dropout=DROPOUT).to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, 'gat_model.pth')
    model.load_state_dict(torch.load(model_path))

    # Visualize attention
    visualize_attention(model, test_loader, DEVICE)


if __name__ == '__main__':
    main()