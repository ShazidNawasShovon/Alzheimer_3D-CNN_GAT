import os
import shutil
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm
import cv2
import glob
import random
from skimage.transform import rotate
from sklearn.model_selection import train_test_split

# Import config variables
from config import *


def load_mri(mri_path):
    """Load MRI image from file."""
    try:
        if mri_path.endswith('.jpg') or mri_path.endswith('.jpeg') or mri_path.endswith('.png'):
            # Load 2D image
            img = cv2.imread(mri_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error loading image: {mri_path}")
                return None
            return img
        else:
            # Load NIfTI file
            img = nib.load(mri_path)
            # Try different methods to get the data array
            try:
                return img.get_fdata()
            except AttributeError:
                try:
                    return img.get_data()
                except AttributeError:
                    return np.asarray(img.dataobj)
    except Exception as e:
        print(f"Error loading MRI from {mri_path}: {e}")
        return None


def normalize_mri(mri):
    """Normalize MRI intensity to [0, 1]."""
    if mri is None:
        return None
    mri = (mri - np.min(mri)) / (np.max(mri) - np.min(mri) + 1e-8)
    return mri


def resize_mri(mri, target_shape):
    """Resize MRI to target shape."""
    if mri is None:
        return None
    # Handle 2D images
    if len(mri.shape) == 2:
        # Resize 2D image
        return cv2.resize(mri, (target_shape[1], target_shape[0]))
    # Handle 3D volumes
    else:
        factors = np.array(target_shape) / np.array(mri.shape)
        return ndimage.zoom(mri, factors, order=1)


def skull_strip(mri):
    """Simple skull stripping using thresholding."""
    if mri is None:
        return None
    mri[mri < 0.1] = 0
    return mri


def augment_mri(mri):
    """Apply data augmentation to MRI."""
    # Random rotation (between -15 and 15 degrees)
    angle = random.uniform(-15, 15)
    if len(mri.shape) == 2:
        # For 2D images
        augmented = rotate(mri, angle, mode='reflect', preserve_range=True)
    else:
        # For 3D volumes, rotate in the axial plane
        augmented = np.zeros_like(mri)
        for i in range(mri.shape[2]):
            augmented[:, :, i] = rotate(mri[:, :, i], angle, mode='reflect', preserve_range=True)
    # Random horizontal flip (50% chance)
    if random.random() > 0.5:
        augmented = np.fliplr(augmented)
    # Random vertical flip (50% chance)
    if random.random() > 0.5:
        augmented = np.flipud(augmented)
    return augmented


def preprocess_mri(mri_path, target_shape, augment=False):
    """Full preprocessing pipeline for a single MRI."""
    mri = load_mri(mri_path)
    if mri is None:
        return None
    # Apply augmentation if requested
    if augment:
        mri = augment_mri(mri)
    mri = normalize_mri(mri)
    mri = resize_mri(mri, target_shape)
    mri = skull_strip(mri)
    return mri


def preprocess_all_mris(raw_data_dir, processed_data_dir, classes, target_shape, augment=False):
    """Preprocess all MRI images in the dataset."""
    for class_name in classes:
        class_dir = os.path.join(raw_data_dir, class_name)
        processed_class_dir = os.path.join(processed_data_dir, class_name)
        os.makedirs(processed_class_dir, exist_ok=True)
        if not os.path.exists(class_dir):
            print(f"Warning: Raw data directory not found: {class_name}")
            continue
        # Find all image files
        file_patterns = ['*.jpg', '*.jpeg', '*.png', '*.nii', '*.nii.gz']
        file_list = []
        for pattern in file_patterns:
            file_list.extend(glob.glob(os.path.join(class_dir, pattern)))
        print(f"Found {len(file_list)} files in {class_name}")
        processed_count = 0
        for file_path in tqdm(file_list, desc=f"Processing {class_name}"):
            file_name = os.path.basename(file_path)
            processed_mri = preprocess_mri(file_path, target_shape, augment=augment)
            if processed_mri is not None:
                # Save processed MRI as NIfTI
                save_path = os.path.join(processed_class_dir, os.path.splitext(file_name)[0] + '.nii.gz')
                # For 2D images, convert to 3D by adding a singleton dimension
                if len(processed_mri.shape) == 2:
                    processed_mri = processed_mri[..., np.newaxis]
                try:
                    nib.save(nib.Nifti1Image(processed_mri, np.eye(4)), save_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error saving processed MRI to {save_path}: {e}")
        print(f"Successfully processed {processed_count} files for {class_name}")


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


def balance_training_data(train_dir, balanced_train_dir, classes, target_shape):
    """
    Balance the training data by oversampling minority classes with augmentation.
    """
    os.makedirs(balanced_train_dir, exist_ok=True)
    # Count files in each class
    class_counts = {}
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith(('.nii', '.nii.gz'))]
        class_counts[class_name] = len(files)

    # Find the class with the maximum number of samples
    max_count = max(class_counts.values())
    print(f"Balancing training data. Maximum class count: {max_count}")

    # For each class, oversample to match the maximum count
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        balanced_class_dir = os.path.join(balanced_train_dir, class_name)
        os.makedirs(balanced_class_dir, exist_ok=True)

        # Get all files for this class
        files = [f for f in os.listdir(class_dir) if f.endswith(('.nii', '.nii.gz'))]

        # Copy original files (without augmentation)
        for file_name in files:
            src = os.path.join(class_dir, file_name)
            dst = os.path.join(balanced_class_dir, file_name)
            shutil.copy2(src, dst)

        # Calculate how many augmented samples we need
        current_count = len(files)
        needed = max_count - current_count
        if needed > 0:
            print(f"Generating {needed} augmented samples for {class_name}")
            # Generate augmented samples
            for i in tqdm(range(needed), desc=f"Augmenting {class_name}"):
                # Randomly select a file to augment
                file_name = random.choice(files)
                file_path = os.path.join(class_dir, file_name)

                # Load the original preprocessed MRI
                img = nib.load(file_path)
                try:
                    mri = img.get_fdata()
                except AttributeError:
                    try:
                        mri = img.get_data()
                    except AttributeError:
                        mri = np.asarray(img.dataobj)

                # Apply augmentation
                augmented_mri = augment_mri(mri)

                # Save augmented sample
                aug_file_name = f"aug_{i}_{file_name}"
                save_path = os.path.join(balanced_class_dir, aug_file_name)
                try:
                    nib.save(nib.Nifti1Image(augmented_mri, np.eye(4)), save_path)
                except Exception as e:
                    print(f"Error saving augmented MRI to {save_path}: {e}")

        final_count = len([f for f in os.listdir(balanced_class_dir) if f.endswith(('.nii', '.nii.gz'))])
        print(f"Class {class_name}: {final_count} samples after balancing")

    print("Training data balancing completed!")


def preprocess_data():
    """Preprocess all data and balance training set."""
    # Create directories
    train_original_dir = os.path.join(PROCESSED_DATA_DIR, 'train_original')
    test_original_dir = os.path.join(PROCESSED_DATA_DIR, 'test_original')
    train_balanced_dir = os.path.join(PROCESSED_DATA_DIR, 'train_balanced')

    # Step 1: Split raw data into train and test
    print("Step 1: Splitting raw data into train and test sets...")
    split_raw_data(RAW_DATA_DIR, train_original_dir, test_original_dir, CLASSES, test_size=0.2)

    # Step 2: Preprocess train and test data
    print("\nStep 2: Preprocessing MRI data...")
    preprocess_all_mris(train_original_dir, train_original_dir, CLASSES, TARGET_SHAPE, augment=False)
    preprocess_all_mris(test_original_dir, test_original_dir, CLASSES, TARGET_SHAPE, augment=False)

    # Step 3: Balance training data
    print("\nStep 3: Balancing training data...")
    balance_training_data(train_original_dir, train_balanced_dir, CLASSES, TARGET_SHAPE)

    print("\nData preprocessing completed!")
    return train_balanced_dir, test_original_dir


if __name__ == "__main__":
    preprocess_data()