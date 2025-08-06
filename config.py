import os
import tarfile
import shutil

# Data paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, 'external')
AAL_ATLAS_TAR = os.path.join(EXTERNAL_DATA_DIR, 'AAL3v2_for_SPM12.tar.gz')
AAL_ATLAS_PATH = os.path.join(EXTERNAL_DATA_DIR, 'AAL3v2.nii')

# Classes
CLASSES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# Preprocessing
TARGET_SHAPE = (128, 128, 128)

# Model parameters
BATCH_SIZE = 4  # Reduced batch size to prevent memory issues
LEARNING_RATE = 0.001
EPOCHS = 10
NUM_GAT_LAYERS = 4
GAT_HIDDEN_DIM = 128
NUM_HEADS = 8
DROPOUT = 0.3

# Feature extraction
USE_CNN_FEATURES = True  # Enable 3D CNN feature extraction

# Model type
MODEL_TYPE = "improved"  # Options: "standard", "improved"

# Training
RANDOM_SEED = 42

# Results
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
MODEL_SAVE_DIR = os.path.join(RESULTS_DIR, 'models')
PLOT_SAVE_DIR = os.path.join(RESULTS_DIR, 'plots')
LOG_SAVE_DIR = os.path.join(RESULTS_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_SAVE_DIR, exist_ok=True)
os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)

# Extract AAL atlas if needed
def extract_aal_atlas():
    """Extract AAL atlas from tar.gz file if not already extracted."""
    if not os.path.exists(AAL_ATLAS_PATH):
        print(f"Extracting AAL atlas from {AAL_ATLAS_TAR}...")
        # Create a temporary directory for extraction
        temp_dir = os.path.join(EXTERNAL_DATA_DIR, 'temp_extract')
        os.makedirs(temp_dir, exist_ok=True)
        try:
            with tarfile.open(AAL_ATLAS_TAR, 'r:gz') as tar:
                # Extract all files to temp directory
                tar.extractall(path=temp_dir)
                # Find the .nii file in the extracted files
                nii_files = []
                for root, _, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith('.nii') or file.endswith('.nii.gz'):
                            nii_files.append(os.path.join(root, file))
                if not nii_files:
                    raise FileNotFoundError("No .nii file found in the atlas archive.")
                # Use the first .nii file found
                source_path = nii_files[0]
                print(f"Found atlas file: {source_path}")
                # Copy to the final location
                shutil.copy2(source_path, AAL_ATLAS_PATH)
                print(f"AAL atlas copied to {AAL_ATLAS_PATH}")
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
    else:
        print(f"AAL atlas already exists at {AAL_ATLAS_PATH}")

# Extract the atlas when config is imported
extract_aal_atlas()

if __name__ == "__main__":
    print("Configuration file executed successfully")