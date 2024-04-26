import os
import random
import shutil

def split_data(source_img_dir, source_label_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=None):
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create destination directories
    train_img_dir = os.path.join(dest_dir, 'train', 'images')
    train_label_dir = os.path.join(dest_dir, 'train', 'labels')
    val_img_dir = os.path.join(dest_dir, 'validation', 'images')
    val_label_dir = os.path.join(dest_dir, 'validation', 'labels')
    test_img_dir = os.path.join(dest_dir, 'test', 'images')
    test_label_dir = os.path.join(dest_dir, 'test', 'labels')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)
    
    # Get list of image file names
    image_files = os.listdir(source_img_dir)
    image_files = [file for file in image_files if file.endswith('.PNG')]
    
    # Shuffle image files to randomize order
    random.shuffle(image_files)
    
    # Split image files into training, validation, and test sets
    num_images = len(image_files)
    num_train_images = int(train_ratio * num_images)
    num_val_images = int(val_ratio * num_images)
    num_test_images = num_images - num_train_images - num_val_images
    
    train_images = image_files[:num_train_images]
    val_images = image_files[num_train_images:num_train_images+num_val_images]
    test_images = image_files[num_train_images+num_val_images:]
    
    # Move image files and corresponding label files to destination directories
    move_files(source_img_dir, train_img_dir, source_label_dir, train_label_dir, train_images)
    move_files(source_img_dir, val_img_dir, source_label_dir, val_label_dir, val_images)
    move_files(source_img_dir, test_img_dir, source_label_dir, test_label_dir, test_images)
    
    print("Data splitting completed successfully.")
    

def move_files(source_img_dir, dest_img_dir, source_label_dir, dest_label_dir, files):
    for file in files:
        img_file = os.path.join(source_img_dir, file)
        label_file = os.path.join(source_label_dir, file.replace('.PNG', '.txt'))
        
        # Move image file
        shutil.copy(img_file, dest_img_dir)
        
        # Move label file
        shutil.copy(label_file, dest_label_dir)
    


def collect_data(train_split_ratio):
    if os.path.isdir("data/train"):
        print("Folder with data already exists.")
        return
    
    print("Copying and splitting files from dataset.")
    source_img_directory = '/datasets/tdt4265/ad/NAPLab-LiDAR/images/'  # Directory containing image files
    source_label_directory = '/datasets/tdt4265/ad/NAPLab-LiDAR/labels_yolo_v1.1/'  # Directory containing label files
    destination_directory = 'data/'  # Directory where split data will be saved

    val_ratio = (1 - train_split_ratio)/2
    split_data(source_img_directory, source_label_directory, destination_directory, train_ratio=train_split_ratio, val_ratio=val_ratio, test_ratio=val_ratio, random_seed=0)

def create_labels(label_dir):
    label_files = os.listdir(label_dir)

