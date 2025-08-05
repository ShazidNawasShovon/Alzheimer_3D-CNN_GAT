import os
import sys
import numpy as np
import torch
from torch_geometric.data import DataLoader as PyGDataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.data_loader import MRIDataset
from models.gat_model import GATModel, GATModelWithAttention
from utils.visualization import plot_attention_weights
from utils.gpu_utils import get_device


def visualize_attention(model_path, test_dir):
    """Visualize attention weights to identify important brain regions."""
    print("=== Attention Visualization ===")

    # Get device
    device = get_device()

    # Create test dataset
    test_dataset = MRIDataset(test_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device)

    # Check if dataset is empty
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. No processed MRI files found.")

    # Create dataloader
    test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    # Create a new model with attention capabilities
    num_node_features = test_dataset[0].x.shape[1]
    attention_model = GATModelWithAttention(
        num_node_features,
        len(CLASSES),
        hidden_dim=GAT_HIDDEN_DIM,
        num_layers=NUM_GAT_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)

    # Load the trained weights
    attention_model.load_state_dict(torch.load(model_path))

    attention_model.eval()

    # Get a batch of data
    data = next(iter(test_loader))
    data = data.to(device)

    # Forward pass with attention weights
    with torch.no_grad():
        _, attention_weights = attention_model(data)

    # Get attention weights from the last layer
    if attention_weights and len(attention_weights) > 0:
        att_weights = attention_weights[-1].cpu().numpy()

        # Average attention across heads
        if len(att_weights.shape) > 1:
            avg_att = np.mean(att_weights, axis=1)
        else:
            avg_att = att_weights

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
        plot_attention_weights(
            node_attention,
            top_n=20,
            save_path=os.path.join(PLOT_SAVE_DIR, 'attention_weights.png')
        )

        # Print top 10 important nodes
        sorted_nodes = sorted(node_attention.items(), key=lambda x: x[1], reverse=True)
        print("Top 10 Important Brain Regions (by attention score):")
        for i, (node_idx, score) in enumerate(sorted_nodes[:10]):
            print(f"{i + 1}. ROI {node_idx}: {score:.4f}")
    else:
        print("No attention weights available for visualization")


if __name__ == "__main__":
    # Check if model exists
    model_path = os.path.join(MODEL_SAVE_DIR, 'gat_model_best.pth')
    if not os.path.exists(model_path):
        print("Trained model not found. Please run train.py first.")
        sys.exit(1)

    # Check if test data exists
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test_original')
    if not os.path.exists(test_dir):
        print("Test data not found. Please run preprocess_data.py first.")
        sys.exit(1)

    # Visualize attention
    visualize_attention(model_path, test_dir)