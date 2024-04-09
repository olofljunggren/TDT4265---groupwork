# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import numpy as np
import gzip
import pickle
import pathlib
from urllib import request
SAVE_PATH = pathlib.Path("notebooks/data/original_lidar")

filename = [
    # ["training_images", "train-images.gz"],
    ["val_images", "val-images.gz"],
    # ["training_labels", "train-labels.gz"],
    ["val_labels", "val-labels.gz"]
]

def extract_lidar():
    save_path = SAVE_PATH.joinpath("lidar.pkl")
    if save_path.is_file():
        return
    
    lidar = {}
    # Load images
    for name in filename[:2]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            print(data.shape)
            lidar[name[0]] = data.reshape((-1,  128*1024))
    # Load labels
    for name in filename[2:]:
        path = SAVE_PATH.joinpath(name[1])
        with gzip.open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
            lidar[name[0]] = data
    with open(save_path, 'wb') as f:
        pickle.dump(lidar, f)


def load():
    extract_lidar()
    dataset_path = SAVE_PATH.joinpath("lidar.pkl")
    with open(dataset_path, 'rb') as f:
        lidar = pickle.load(f)
    X_train, Y_train, X_test, Y_test = lidar["training_images"], lidar["training_labels"], lidar["test_images"], lidar["test_labels"]
    return X_train.reshape(-1, 128, 1024), Y_train, X_test.reshape(-1, 128, 1024), Y_test

