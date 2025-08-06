import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import *
from scripts.preprocess_data import preprocess_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.visualize_attention import visualize_attention
from utils.gpu_utils import check_gpu_availability


def main():
    """Main function to run the entire pipeline."""
    print("=== Alzheimer's Disease Detection using GATs ===")
    # Check GPU availability
    check_gpu_availability()

    # Step 1: Preprocess data
    print("\nStep 1: Preprocessing data...")
    train_dir, test_dir = preprocess_data()
    print(f"Data preprocessed. Train dir: {train_dir}, Test dir: {test_dir}")

    # Step 2: Train model with CNN features enabled
    print("\nStep 2: Training model...")
    model_path = train_model(train_dir, test_dir, use_cnn_features=USE_CNN_FEATURES)

    # Check if model was trained successfully
    if not model_path or not os.path.exists(model_path):
        print("Error: Model training failed or model file not found.")
        return

    # Step 3: Evaluate model
    print("\nStep 3: Evaluating model...")
    model, preds, labels = evaluate_model(model_path, test_dir, use_cnn_features=USE_CNN_FEATURES)

    if model is None:
        print("Error: Model evaluation failed.")
        return

    # Step 4: Visualize attention
    print("\nStep 4: Visualizing attention weights...")
    visualize_attention(model_path, test_dir, use_cnn_features=USE_CNN_FEATURES)

    print("\n=== Pipeline completed successfully! ===")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()