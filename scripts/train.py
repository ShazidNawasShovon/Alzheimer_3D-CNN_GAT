import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils.data_loader import MRIDataset, get_class_weights
from models.gat_model import GATModel
from utils.visualization import plot_training_history
from utils.gpu_utils import get_device


def train_model(train_dir, test_dir, use_cnn_features=False):
    """Train the GAT model."""
    print("=== Model Training ===")
    print(f"Using CNN features: {use_cnn_features}")
    # Get device
    device = get_device()
    # Create datasets
    train_dataset = MRIDataset(train_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device, use_cnn_features)
    test_dataset = MRIDataset(test_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device, use_cnn_features)
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty. No processed MRI files found.")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty. No processed MRI files found.")
    # Create dataloaders using PyG's DataLoader
    train_loader = PyGDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = PyGDataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    # Get class weights
    class_weights = get_class_weights(train_dataset).to(device)
    # Initialize model
    num_node_features = train_dataset[0].x.shape[1]
    model = GATModel(
        num_node_features,
        len(CLASSES),
        hidden_dim=GAT_HIDDEN_DIM,
        num_layers=NUM_GAT_LAYERS,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(device)
    # Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Training loop
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_f1_scores = []
    best_test_acc = 0.0
    best_epoch = 0
    patience = 10  # Early stopping patience
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Training]"):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.num_graphs

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Validation]"):
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                test_loss += loss.item() * batch.num_graphs
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())

        epoch_test_loss = test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)

        # Calculate metrics
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='weighted')
        test_accuracies.append(test_acc)
        test_f1_scores.append(test_f1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}: Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_test_loss:.4f}, Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0
            # Save model
            model_path = os.path.join(MODEL_SAVE_DIR, 'gat_model_best.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with accuracy {best_test_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Plot training history
    plot_training_history(train_losses, test_losses, test_accuracies, test_f1_scores,
                          os.path.join(PLOT_SAVE_DIR, 'training_history.png'))

    print(f"Training completed. Best model at epoch {best_epoch} with accuracy {best_test_acc:.4f}")

    # Return the path to the best model
    return os.path.join(MODEL_SAVE_DIR, 'gat_model_best.pth')


if __name__ == "__main__":
    # Check if processed data exists
    train_dir = os.path.join(PROCESSED_DATA_DIR, 'train_balanced')
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test_original')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print("Processed data not found. Please run preprocess_data.py first.")
        sys.exit(1)

    # Train model
    model_path = train_model(train_dir, test_dir)
    print(f"Model saved to: {model_path}")