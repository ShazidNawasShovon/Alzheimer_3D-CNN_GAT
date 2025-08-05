import numpy as np
import torch
import nibabel as nib
from scipy import ndimage
import os
import cupy as cp


def load_atlas(atlas_path):
    """Load the AAL atlas."""
    try:
        # Try to load the atlas file
        atlas_img = nib.load(atlas_path)
        # Try different methods to get the data array
        try:
            return atlas_img.get_fdata()
        except AttributeError:
            try:
                return atlas_img.get_data()
            except AttributeError:
                return np.asarray(atlas_img.dataobj)
    except Exception as e:
        print(f"Error loading atlas: {e}")
        print(f"Atlas path: {atlas_path}")
        print(f"File exists: {os.path.exists(atlas_path)}")
        if os.path.exists(atlas_path):
            print(f"File size: {os.path.getsize(atlas_path)} bytes")
        raise


def extract_roi_features(mri, atlas, roi_values):
    """Extract features from each ROI in the MRI."""
    roi_features = []

    # Ensure atlas and MRI have the same shape
    if atlas.shape != mri.shape:
        print(f"Resizing atlas from {atlas.shape} to match MRI shape {mri.shape}")
        # Resize atlas to match MRI
        factors = np.array(mri.shape) / np.array(atlas.shape)
        atlas = ndimage.zoom(atlas, factors, order=0)

    for roi_val in roi_values:
        # Create ROI mask
        roi_mask = (atlas == roi_val)

        # Extract ROI region
        roi_region = mri[roi_mask]

        if len(roi_region) > 0:
            # Basic features: mean intensity, std, and texture measures
            mean_intensity = np.mean(roi_region)
            std_intensity = np.std(roi_region)

            # Texture measure - handle small regions
            if len(roi_region) < 2:
                # If ROI has only one voxel, set texture to 0
                texture = 0.0
            else:
                try:
                    # For 1D array (flattened ROI)
                    if len(roi_region.shape) == 1:
                        grad = np.gradient(roi_region)
                        grad_mag = np.mean(np.abs(grad))
                    else:
                        # For 2D or 3D ROI
                        grad = np.gradient(roi_region)
                        if isinstance(grad, tuple):
                            grad_mag = np.sqrt(sum(g ** 2 for g in grad))
                        else:
                            grad_mag = np.abs(grad)
                        grad_mag = np.mean(grad_mag)
                    texture = float(grad_mag)
                except Exception as e:
                    print(f"Error computing gradient for ROI {roi_val}: {e}")
                    texture = 0.0

            roi_features.append([mean_intensity, std_intensity, texture])
        else:
            roi_features.append([0, 0, 0])

    return np.array(roi_features)


def create_graph_edges(atlas, roi_values, threshold=None):
    """Create graph edges based on ROI adjacency in the atlas."""
    num_rois = len(roi_values)
    adj_matrix = np.zeros((num_rois, num_rois))

    # Get centroids of each ROI
    centroids = []
    for roi_val in roi_values:
        roi_mask = (atlas == roi_val)
        coords = np.argwhere(roi_mask)
        if len(coords) > 0:
            centroid = np.mean(coords, axis=0)
            centroids.append(centroid)
        else:
            centroids.append(np.zeros(3))

    centroids = np.array(centroids)

    # If threshold is not provided, use a fraction of the maximum distance
    if threshold is None:
        max_dist = np.max([np.linalg.norm(centroids[i] - centroids[j])
                           for i in range(num_rois) for j in range(i + 1, num_rois)])
        threshold = max_dist * 0.2  # Connect ROIs within 20% of max distance

    # Connect ROIs if centroids are within the threshold distance
    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            if dist < threshold:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1

    # Convert to edge index format for PyG
    edge_index = np.argwhere(adj_matrix > 0).T
    return torch.tensor(edge_index, dtype=torch.long)


if __name__ == "__main__":
    from config import *

    # Test graph utilities
    print("Testing graph utilities...")

    # Load atlas
    atlas = load_atlas(AAL_ATLAS_PATH)
    print(f"Atlas loaded successfully. Shape: {atlas.shape}")

    # Get ROI values
    roi_values = np.unique(atlas)[1:]  # Exclude background (0)
    print(f"Number of ROIs: {len(roi_values)}")

    # Create a dummy MRI
    dummy_mri = np.random.rand(*atlas.shape)

    # Extract ROI features
    roi_features = extract_roi_features(dummy_mri, atlas, roi_values)
    print(f"ROI features extracted successfully. Shape: {roi_features.shape}")

    # Create graph edges
    edge_index = create_graph_edges(atlas, roi_values)
    print(f"Graph edges created successfully. Shape: {edge_index.shape}")

    print("Graph utilities test completed successfully!")