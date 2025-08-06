import os
import sys
import numpy as np
import torch
from torch_geometric.data import DataLoader as PyGDataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils.data_loader import MRIDataset
from models.improved_gat_model import ImprovedGATModel
from utils.visualization import plot_confusion_matrix
from utils.gpu_utils import get_device


def evaluate_model(model_path, test_dir, use_cnn_features=False):
    """Evaluate the trained model."""
    print("=== Model Evaluation ===")
    print(f"Using CNN features: {use_cnn_features}")
    print(f"Using model type: {MODEL_TYPE}")

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you have trained the model first.")
        return None, [], []

    # Get device
    device = get_device()

    # Create test dataset
    test_dataset = MRIDataset(test_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device, use_cnn_features)

    # Check if dataset is empty
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. No processed MRI files found.")

    # Create dataloader
    test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Test dataset size: {len(test_dataset)}")

    # Initialize model based on model type
    num_node_features = test_dataset[0].x.shape[1]

    if MODEL_TYPE == "improved":
        model = ImprovedGATModel(
            num_node_features,
            len(CLASSES),
            hidden_dim=GAT_HIDDEN_DIM,
            num_layers=NUM_GAT_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        ).to(device)
    else:
        from models.gat_model import GATModel
        model = GATModel(
            num_node_features,
            len(CLASSES),
            hidden_dim=GAT_HIDDEN_DIM,
            num_layers=NUM_GAT_LAYERS,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        ).to(device)

    # Load model weights with error handling
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, [], []

    # Evaluate model
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = batch.to(device)
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate AUC for each class (one-vs-rest)
    aucs = []
    for i in range(len(CLASSES)):
        binary_labels = np.array([1 if label == i else 0 for label in all_labels])
        binary_probs = np.array([prob[i] for prob in all_probs])
        if len(np.unique(binary_labels)) > 1:
            auc = roc_auc_score(binary_labels, binary_probs)
            aucs.append(auc)
        else:
            aucs.append(0.5)
    mean_auc = np.mean(aucs)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Test Mean AUC: {mean_auc:.4f}")

    # Plot confusion matrix
    plot_confusion_matrix(cm, CLASSES, os.path.join(PLOT_SAVE_DIR, f'{MODEL_TYPE}_confusion_matrix.png'))

    return model, all_preds, all_labels


if __name__ == "__main__":
    # Check if model exists
    model_path = os.path.join(MODEL_SAVE_DIR, f'{MODEL_TYPE}_gat_model_best.pth')
    if not os.path.exists(model_path):
        print("Trained model not found. Please run train.py first.")
        sys.exit(1)
    # Check if test data exists
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test_original')
    if not os.path.exists(test_dir):
        print("Test data not found. Please run preprocess_data.py first.")
        sys.exit(1)
    # Evaluate model
    model, preds, labels = evaluate_model(model_path, test_dir, use_cnn_features=USE_CNN_FEATURES)