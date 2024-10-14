# Deepfake-Detection-CNN

This project uses a Convolutional Neural Network (CNN) to detect deepfake images and frames of vdieos.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- Kaggle API (for dataset download)

## Setup

1. Clone this repository:

```
git clone https://github.com/sia1408/Deepfake-Detection-CNN.git
cd Deepfake-Detection-CNN
```

2. Set up Kaggle api then install dependencies:

```
mkdir -p ~/.kaggle
cp config/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

```
pip install -r requirements.txt
```

3. Download the dataset using Kaggle API:
   
```
python data/kaggle_download.py
```
4. Preprocess images
```
python scripts/image_preprocessing.py
```
5. Train the model:
```
python scripts/train.py
```

## Project Structure

Deepfake-Detection-CNN/
│
├── data/
│   ├── kaggle_download.py
│   ├── faces_224/
│   └── metadata.csv
│
├── models/
│   ├── cnn_model.py
│   ├── model_checkpoint.h5
│   └── final_model.h5
│
├── scripts/
│   ├── train.py
│   ├── image_preprocessing.py
│   └── other_scripts.py
│
├── config/
│   └── kaggle.json
│
├── requirements.txt
│
└── README.md
