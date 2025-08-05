import os
import sys
import shutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from utils.preprocessing import preprocess_all_mris, balance_training_data
from utils.data_split import split_raw_data
from utils.gpu_utils import get_device


def preprocess_data():
    """Split raw data into train/test and preprocess both sets."""
    print("=== Data Preprocessing ===")

    # Get device
    device = get_device()

    # Create temporary directories for raw train and test data
    raw_train_dir = os.path.join(DATA_DIR, 'raw_train')
    raw_test_dir = os.path.join(DATA_DIR, 'raw_test')

    # Split the raw data
    print("Splitting raw data into train and test sets...")
    split_raw_data(RAW_DATA_DIR, raw_train_dir, raw_test_dir, CLASSES, test_size=0.2, random_seed=RANDOM_SEED)

    # Preprocess training data WITHOUT augmentation first (original form)
    print("\nPreprocessing training data (original form)...")
    processed_train_dir = os.path.join(PROCESSED_DATA_DIR, 'train_original')
    preprocess_all_mris(raw_train_dir, processed_train_dir, CLASSES, TARGET_SHAPE, augment=False)

    # Balance training data by generating augmented samples
    print("\nBalancing training data with augmentation...")
    balanced_train_dir = os.path.join(PROCESSED_DATA_DIR, 'balanced_train')
    balance_training_data(processed_train_dir, balanced_train_dir, CLASSES, TARGET_SHAPE)

    # Preprocess test data WITHOUT augmentation (keep original form)
    print("\nPreprocessing test data (original form)...")
    processed_test_dir = os.path.join(PROCESSED_DATA_DIR, 'test_original')
    preprocess_all_mris(raw_test_dir, processed_test_dir, CLASSES, TARGET_SHAPE, augment=False)

    # Clean up temporary directories
    shutil.rmtree(raw_train_dir)
    shutil.rmtree(raw_test_dir)

    print("\nData preprocessing completed successfully!")
    return balanced_train_dir, processed_test_dir


if __name__ == "__main__":
    preprocess_data()