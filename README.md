# Alzheimer's Disease Detection using Graph Attention Networks

This project implements a deep learning pipeline for early detection of Alzheimer's Disease using Graph Attention Networks (GATs) on structural MRI data.

## Project Structure

## Python version `3.12` will be needed for the project.
## Setup
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone 'url will coming soon'
   ```
2. Inside the project folder open `powershell` and create a virtual environment named 'venv':
   ```powershell
   python -m venv venv
   ```
3. Activate the virtual environment:
   ```powershell
   .\venv\Scripts\Activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
5. After install the required packages, go to the **`.venv/Scripts`** path:
   ```bash
   cd .venv/Scripts
   ```
6. Now, install the **`torchaudio --index-url https://download.pytorch.org/whl/cu118`**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
7. First, check GPU availability   
   ```bash
   python utils/gpu_utils.py
   ```
8. Test configuration and extract AAL atlas   
   ```bash
   python config.py
   ```
9. Test preprocessing functions   
   ```bash
   python utils/preprocessing.py
   ```
10. Test data splitting   
   ```bash
   python utils/data_split.py
   ```
11. Test graph utilities   
   ```bash
   python utils/graph_utils.py
   ```
12. Test GAT model   
   ```bash
   python models/gat_model.py
   ```
13. Test visualization functions   
   ```bash
   python utils/visualization.py
   ```
14. Preprocess the data (split raw data, augment training data, keep test data original)   
   ```bash
   python scripts/preprocess_data.py
   ```
15. Train the GAT model   
   ```bash
   python scripts/train.py
   ```
16. Evaluate the trained model   
   ```bash
   python scripts/evaluate.py
   ```
17. Visualize attention weights   
   ```bash
   python scripts/visualize_attention.py
   ```
18. Run the entire pipeline in one go (optional)
   ```bash
   python main.py
   ```
   
   