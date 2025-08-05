import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_training_history(train_losses, test_losses, test_accuracies, test_f1_scores, save_path):
    """Plot training history including loss and metrics."""
    plt.figure(figsize=(15, 5))

    # Plot training and test loss
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot test accuracy
    plt.subplot(1, 3, 2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot test F1 score
    plt.subplot(1, 3, 3)
    plt.plot(test_f1_scores, label='Test F1 Score')
    plt.title('Test F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(cm, classes, save_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()


def plot_attention_weights(node_attention, top_n=20, save_path='attention_weights.png'):
    """Plot attention weights for top brain regions."""
    # Sort nodes by attention score
    sorted_nodes = sorted(node_attention.items(), key=lambda x: x[1], reverse=True)

    # Get top nodes and scores
    nodes = [f"ROI {node}" for node, _ in sorted_nodes[:top_n]]
    scores = [score for _, score in sorted_nodes[:top_n]]

    # Create horizontal bar plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(nodes)), scores)
    plt.yticks(range(len(nodes)), nodes)
    plt.xlabel('Attention Score')
    plt.title(f'Top {top_n} Important Brain Regions')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    from config import *

    # Test visualization functions
    print("Testing visualization functions...")

    # Create dummy data
    train_losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    test_losses = [0.6, 0.5, 0.4, 0.3, 0.2]
    test_accuracies = [0.7, 0.8, 0.85, 0.9, 0.92]
    test_f1_scores = [0.65, 0.75, 0.8, 0.85, 0.9]

    # Test plot_training_history
    plot_training_history(train_losses, test_losses, test_accuracies, test_f1_scores,
                          os.path.join(PLOT_SAVE_DIR, 'test_training_history.png'))
    print("Training history plot saved successfully!")

    # Test plot_confusion_matrix
    cm = np.array([[50, 5, 2, 3], [10, 40, 5, 5], [5, 10, 35, 10], [2, 3, 5, 40]])
    plot_confusion_matrix(cm, CLASSES, os.path.join(PLOT_SAVE_DIR, 'test_confusion_matrix.png'))
    print("Confusion matrix plot saved successfully!")

    # Test plot_attention_weights
    node_attention = {i: np.random.rand() for i in range(50)}
    plot_attention_weights(node_attention, top_n=20,
                           save_path=os.path.join(PLOT_SAVE_DIR, 'test_attention_weights.png'))
    print("Attention weights plot saved successfully!")

    print("Visualization functions test completed successfully!")