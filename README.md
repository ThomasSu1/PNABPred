# Getting Started
## Installation
```sh
# 1. Create a new conda environment
$conda create --name protein_pred python=3.9
$ conda activate protein_pred

# 2. Install dependencies
# It's recommended to install PyTorch first, matching your CUDA version.
# Visit [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for specific instructions.

# 3. Install the remaining packages from requirements.txt
$ pip install -r requirements.txt

# 4. Clone this repository
$ git clone https://github.com/ThomasSu1/PNABPred.git
```
# Usage
To train a new model, run `train.py`. The training log will be saved to `model_training.log`.
```sh
# Navigate to the project directory
$ cd your-repo-name

# Run the training script
$ python train.py
```
# Acknowledgement
- ESM: https://github.com/facebookresearch/esm
