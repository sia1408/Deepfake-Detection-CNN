import os
from kaggle.api.kaggle_api_extended import KaggleApi

def download_data():
    os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), 'config')
    api = KaggleApi()
    api.authenticate()

    dataset = 'dagnelies/deepfake-faces'
    output_path = 'data/'
    api.dataset_download_files(dataset, path=output_path, unzip=True)

if __name__ == "__main__":
    download_data()