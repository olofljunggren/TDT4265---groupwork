# Code modified from: https://github.com/hsjeong5/MNIST-for-Numpy
import numpy as np
import gzip
import pickle
import pathlib
from urllib import request
import cv2 as cv
import os
import json
SAVE_PATH = pathlib.Path("notebooks/data/original_lidar")

dictnames = [
    ["training_images"],
    ["val_images"],
    ["training_labels"],
    ["val_labels"]
]

def get_image_size(url):
    im = np.array(cv.imread(url, cv.IMREAD_GRAYSCALE))
    return im.shape

def extract_lidar():
    save_path = SAVE_PATH.joinpath("lidar.pkl")
    if save_path.is_file():
        return
    
    lidar = {}
    # Load images
    base_url = "/datasets/tdt4265/ad/NAPLab-LiDAR/"
    with open(base_url+"train.txt", 'r') as file:
        filenames = [line.rstrip() for line in file]

    training_samples = 1800
    total_samples = len(filenames)

    size = get_image_size(base_url+"images/frame_000001.PNG")

    # Save train images
    train_data = np.empty((training_samples, size[0]*size[1]))
    # Iterate through each PNG image file in the directory
    for index, filename in enumerate(filenames[:training_samples]):
        im = np.array(cv.imread(base_url+filename, cv.IMREAD_GRAYSCALE))
        train_data[index,:] = im 
    lidar[dictnames[0]] = train_data

    # Save validation images
    val_data = np.empty((total_samples-training_samples, size[0]*size[1]))
    # Iterate through each PNG image file in the directory
    for index, filename in enumerate(filenames[training_samples:total_samples]):
        im = np.array(cv.imread(base_url+filename, cv.IMREAD_GRAYSCALE))
        val_data[index,:] = im 
    lidar[dictnames[1]] = val_data

    
    # Base URL where the label files are stored
    base_url = "labels_yolo_v1.1/"

    # Save train labels
    annotations = []
    # Loop through each frame
    for i in range(training_samples):
        number = str(i).zfill(6)  # Format frame number
        filename = f"frame_{number}.txt"  # Assuming file names are like "frame_000001.txt"

        # Check if the file exists
        if os.path.exists(base_url + filename):
            # Load annotations from the text file
            labels = np.loadtxt(base_url + filename)

            # Prepare annotations for the current frame
            frame_annotations = []
            for label in labels:
                obj_class, x_center, y_center, width, height = label
                annotation = {"class": int(obj_class), "x_center": float(x_center),
                            "y_center": float(y_center), "width": float(width),
                            "height": float(height)}
                frame_annotations.append(annotation)

            # Add annotations for the current frame to the list
            annotations.append({"image_path": f"path/to/frames/frame_{number}.jpg",
                                "annotations": frame_annotations})
    lidar[dictnames[2]] = annotations

    # Save validation labels
    annotations = []
    # Loop through each frame
    for i in range(training_samples,total_samples):
        number = str(i).zfill(6)  # Format frame number
        filename = f"frame_{number}.txt"  # Assuming file names are like "frame_000001.txt"

        # Check if the file exists
        if os.path.exists(base_url + filename):
            # Load annotations from the text file
            labels = np.loadtxt(base_url + filename)

            # Prepare annotations for the current frame
            frame_annotations = []
            for label in labels:
                obj_class, x_center, y_center, width, height = label
                annotation = {"class": int(obj_class), "x_center": float(x_center),
                            "y_center": float(y_center), "width": float(width),
                            "height": float(height)}
                frame_annotations.append(annotation)

            # Add annotations for the current frame to the list
            annotations.append({"image_path": f"path/to/frames/frame_{number}.jpg",
                                "annotations": frame_annotations})
    lidar[dictnames[3]] = annotations

    with open(save_path, 'wb') as f:
        pickle.dump(lidar, f)


def load():
    extract_lidar()
    dataset_path = SAVE_PATH.joinpath("lidar.pkl")

    # Get image size
    base_url = "/datasets/tdt4265/ad/NAPLab-LiDAR/"
    height, width = get_image_size(base_url+"images/frame_000001.PNG")
    with open(dataset_path, 'rb') as f:
        lidar = pickle.load(f)
    X_train, Y_train, X_test, Y_test = lidar["training_images"], lidar["training_labels"], lidar["test_images"], lidar["test_labels"]
    return X_train.reshape(-1, height, width), Y_train, X_test.reshape(-1, height, width), Y_test


