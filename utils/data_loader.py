import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.utils.class_weight import compute_class_weight
from .graph_utils import load_atlas, extract_roi_features, create_graph_edges
from models.feature_extractor import FeatureExtractor3D


class MRIDataset(Dataset):
    def __init__(self, processed_data_dir, classes, class_to_idx, atlas_path, target_shape, device='cpu',
                 use_cnn_features=False):
        self.processed_data_dir = processed_data_dir
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.atlas = load_atlas(atlas_path)
        self.roi_values = np.unique(self.atlas)[1:]  # Exclude background (0)
        self.num_rois = len(self.roi_values)
        self.target_shape = target_shape
        self.device = device
        self.use_cnn_features = use_cnn_features

        # Initialize feature extractor if using CNN features
        if use_cnn_features:
            self.feature_extractor = FeatureExtractor3D(in_channels=1, feature_dim=64).to(device)
            self.feature_extractor.eval()  # Set to evaluation mode

        # Load file paths and labels
        self.file_paths = []
        self.labels = []
        print(f"Looking for processed data in: {processed_data_dir}")
        print(f"Using CNN features: {use_cnn_features}")

        for class_name in classes:
            class_dir = os.path.join(processed_data_dir, class_name)
            print(f"Checking class directory: {class_name}")
            if not os.path.exists(class_dir):
                print(f"Warning: Directory not found: {class_name}")
                continue
            file_list = os.listdir(class_dir)
            print(f"Found {len(file_list)} files in {class_name}")
            for file_name in file_list:
                if file_name.endswith('.nii') or file_name.endswith('.nii.gz'):
                    file_path = os.path.join(class_dir, file_name)
                    self.file_paths.append(file_path)
                    self.labels.append(class_to_idx[class_name])

        print(f"Total files found: {len(self.file_paths)}")
        if len(self.file_paths) == 0:
            raise ValueError("No MRI files found in the processed data directory. Please check the preprocessing step.")

        # Create graph edges (same for all subjects)
        self.edge_index = create_graph_edges(self.atlas, self.roi_values)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load MRI
        mri_path = self.file_paths[idx]
        img = nib.load(mri_path)

        # Try different methods to get the data array
        try:
            mri = img.get_fdata()
        except AttributeError:
            try:
                mri = img.get_data()
            except AttributeError:
                mri = np.asarray(img.dataobj)

        # If the MRI is 2D (with singleton dimension), expand to 3D by repeating
        if len(mri.shape) == 3 and mri.shape[2] == 1:
            mri = np.repeat(mri, self.target_shape[2], axis=2)

        # Extract features based on the method
        if self.use_cnn_features:
            # Use CNN to extract features
            # Convert to tensor and add batch and channel dimensions
            mri_tensor = torch.tensor(mri, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(self.device)

            # Extract features using CNN
            with torch.no_grad():
                try:
                    cnn_features = self.feature_extractor(mri_tensor).cpu().numpy()

                    # Reshape features to match ROI structure
                    # We'll distribute the CNN features across ROIs
                    num_rois = len(self.roi_values)
                    features_per_roi = cnn_features.shape[1] // num_rois

                    # If we can't evenly distribute, pad with zeros
                    if cnn_features.shape[1] < num_rois * features_per_roi:
                        padding = np.zeros((1, num_rois * features_per_roi - cnn_features.shape[1]))
                        cnn_features = np.concatenate([cnn_features, padding], axis=1)

                    # Reshape to (num_rois, features_per_roi)
                    roi_features = cnn_features.reshape(num_rois, features_per_roi)
                except Exception as e:
                    print(f"Error extracting CNN features: {e}")
                    print("Falling back to ROI-based features")
                    roi_features = extract_roi_features(mri, self.atlas, self.roi_values)
        else:
            # Use traditional ROI-based features
            roi_features = extract_roi_features(mri, self.atlas, self.roi_values)

        # Create graph data object
        data = Data(
            x=torch.tensor(roi_features, dtype=torch.float).to(self.device),
            edge_index=self.edge_index.to(self.device),
            y=torch.tensor(self.labels[idx], dtype=torch.long).to(self.device)
        )
        return data


def get_class_weights(dataset):
    """Compute class weights for balanced training."""
    try:
        labels = [data.y.item() for data in dataset]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        return torch.tensor(class_weights, dtype=torch.float)
    except Exception as e:
        print(f"Error computing class weights: {e}")
        # Return equal weights as fallback
        return torch.ones(len(np.unique([data.y.item() for data in dataset])), dtype=torch.float)


if __name__ == "__main__":
    from config import *
    from ..utils.gpu_utils import get_device

    # Test data loader
    print("Testing data loader...")
    # Get device
    device = get_device()
    # Create a test dataset
    test_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
    if os.path.exists(test_dir):
        # Test with traditional ROI features
        print("\nTesting with traditional ROI features:")
        dataset_roi = MRIDataset(test_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device,
                                 use_cnn_features=False)
        print(f"Dataset created successfully. Size: {len(dataset_roi)}")
        # Test getting a sample
        sample = dataset_roi[0]
        print(f"Sample data: {sample}")
        print(f"Sample features shape: {sample.x.shape}")
        print(f"Sample label: {sample.y}")
        # Test with CNN features
        print("\nTesting with CNN features:")
        dataset_cnn = MRIDataset(test_dir, CLASSES, CLASS_TO_IDX, AAL_ATLAS_PATH, TARGET_SHAPE, device,
                                 use_cnn_features=True)
        print(f"Dataset created successfully. Size: {len(dataset_cnn)}")
        # Test getting a sample
        sample = dataset_cnn[0]
        print(f"Sample data: {sample}")
        print(f"Sample features shape: {sample.x.shape}")
        print(f"Sample label: {sample.y}")
        # Test class weights
        class_weights = get_class_weights(dataset_cnn)
        print(f"Class weights: {class_weights}")
    else:
        print("Test directory not found. Skipping data loader test.")