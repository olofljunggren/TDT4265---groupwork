import gzip
import numpy as np
import cv2 
import pathlib

SAVE_PATH = pathlib.Path("data/original_lidar")
base_url = "/datasets/tdt4265/ad/NAPLab-LiDAR/"
with open(base_url+"train.txt", 'r') as file:
    filenames = [line.rstrip() for line in file]

training_samples = 1800
total_samples = len(filenames)

# Save training images
filepath = SAVE_PATH.joinpath("train-images.gz")
if not filepath.is_file():
    with gzip.open(filepath, 'ab') as gz_file:
        # Iterate through each PNG image file in the directory
        for index, filename in enumerate(filenames[:training_samples]):
            im = np.array(cv2.imread(base_url+filename, cv2.IMREAD_GRAYSCALE))
            
            # Convert the image to bytes and write it to the gzipped file
            img_bytes = im.tobytes()
            gz_file.write(img_bytes)
    print("Training image zip saved.")
else:
    print("Training image zip already exists.")


# Save validation images
filepath = SAVE_PATH.joinpath("val-images.gz")
if not filepath.is_file():
    with gzip.open(filepath, 'ab') as gz_file:
        # Iterate through each PNG image file in the directory
        for index, filename in enumerate(filenames[training_samples:]):
            im = np.array(cv2.imread(base_url+filename, cv2.IMREAD_GRAYSCALE))
            
            # Convert the image to bytes and write it to the gzipped file
            img_bytes = im.tobytes()
            gz_file.write(img_bytes)
    print("Val image zip saved.")
else:
    print("Val image zip already exists.")


# Save training labels
filepath = SAVE_PATH.joinpath("train-labels.gz")
if not filepath.is_file():
    with gzip.open(filepath, 'ab') as gz_file:
        # Iterate through each PNG image file in the directory
        for i in range(training_samples):
            number = str(i).zfill(6)
            filename = f"labels_yolo_v1.1/frame_{number}.txt"
            labels = np.loadtxt(base_url+filename)
            
            # Convert the image to bytes and write it to the gzipped file
            img_bytes = im.tobytes()
            gz_file.write(img_bytes)
    print("Training label zip saved.")
else:
    print("Training label zip already exists.")


# Save validation labels
filepath = SAVE_PATH.joinpath("val-labels.gz")
if not filepath.is_file():
    with gzip.open(filepath, 'ab') as gz_file:
        # Iterate through each PNG image file in the directory
        for i in range(training_samples, total_samples):
            number = str(i).zfill(6)
            filename = f"labels_yolo_v1.1/frame_{number}.txt"
            labels = np.loadtxt(base_url+filename)
            
            # Convert the image to bytes and write it to the gzipped file
            img_bytes = im.tobytes()
            gz_file.write(img_bytes)
    print("Val label zip saved.")
else:
    print("Val label zip already exists.")

# Find shape of images
im = np.array(cv2.imread(base_url+filenames[0], cv2.IMREAD_GRAYSCALE))
print("Input shape = ", im.shape)