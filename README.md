# Deepfake-Detection-CNN

This project uses a Convolutional Neural Network (CNN) to detect deepfake images and frames of vdieos.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Kaggle API (for dataset download)

## Setup

1. Clone this repository:

git clone https://github.com/sia1408/deepfake-detection-project.git
cd deepfake-detection

2. Install dependencies:

pip install -r requirements.txt


3. Download the dataset using Kaggle API:

python kaggle_download.py

4. Train the model:

python train.py


## Project Structure

- `data/`
- `models/`
- `scripts/`
- `utils/`