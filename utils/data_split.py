import os
import shutil
from sklearn.model_selection import train_test_split
import random


def split_raw_data(raw_data_dir, train_dir, test_dir, classes, test_size=0.2, random_seed=42):
    """
    Split raw data into train and test sets.

    Args:
        raw_data_dir: Path to the raw data directory
        train_dir: Path to save the training data
        test_dir: Path to save the test data
        classes: List of class names
        test_size: Proportion of data to use for testing
        random_seed: Random seed for reproducibility
    """
    # Set random seeds
    random.seed(random_seed)

    # Create train and test directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # For each class, split the files
    for class_name in classes:
        class_dir = os.path.join(raw_data_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        # Create class directories
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # Get all files for this class
        files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.nii', '.nii.gz'))]

        # Split into train and test
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_seed)

        # Copy files to train directory
        for file_name in train_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(train_class_dir, file_name)
            shutil.copy2(src, dst)

        # Copy files to test directory
        for file_name in test_files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(test_class_dir, file_name)
            shutil.copy2(src, dst)

        print(f"Class {class_name}: {len(train_files)} train files, {len(test_files)} test files")

    print("Data split completed!")


if __name__ == "__main__":
    from config import *

    # Test data splitting
    print("Testing data splitting...")

    # Create test directories
    test_raw_dir = os.path.join(DATA_DIR, 'test_raw')
    test_train_dir = os.path.join(DATA_DIR, 'test_train')
    test_test_dir = os.path.join(DATA_DIR, 'test_test')

    # Create test class directories
    for class_name in CLASSES:
        class_dir = os.path.join(test_raw_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Create dummy files
        for i in range(10):
            with open(os.path.join(class_dir, f"file_{i}.jpg"), 'w') as f:
                f.write("dummy content")

    # Test splitting
    split_raw_data(test_raw_dir, test_train_dir, test_test_dir, CLASSES, test_size=0.2)

    # Clean up
    shutil.rmtree(test_raw_dir)
    shutil.rmtree(test_train_dir)
    shutil.rmtree(test_test_dir)

    print("Data splitting test completed successfully!")