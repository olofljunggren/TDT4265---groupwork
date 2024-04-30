import os
import random
import shutil
import numpy as np

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
    for filename in files:
        img_file = os.path.join(source_img_dir, filename)
        label_file = os.path.join(source_label_dir, filename.replace('.PNG', '.txt'))
        
        # Move image file
        shutil.copy(img_file, dest_img_dir)
        
        # Image format hardcoded 
        image_width = 1024
        image_height = 128

        # Open label file and convert to SSD format
        converted_labels = []
        labels = np.loadtxt(label_file)

        for row in labels:
            try:
                label, x_middle, y_middle, x_width, y_height = row
            except Exception:
                label, x_middle, y_middle, x_width, y_height = labels
            x_min = max(0, (x_middle - x_width / 2))
            x_max = min(image_width, (x_middle + x_width / 2))
            y_min = max(0, (y_middle - y_height / 2))
            y_max = min(image_height, (y_middle + y_height / 2))
            converted_labels.append((label, x_min, x_max, y_min, y_max))

        for label in converted_labels:
            xml_output = generate_xml(labels, filename, image_width, image_height)
        # Move label file
        
        # np.savetxt(dest_label_dir+"/"+label_file.split("/")[-1], np.array(converted_labels))
        output_xml_file = dest_label_dir + "/" + label_file.split("/")[-1] + '.xml'
        with open(output_xml_file, 'w') as xml_file:
            xml_file.write(xml_output)

def generate_xml(labels, image_filename, image_width, image_height):
    xml_content = f'''<annotation>
    <folder>data</folder>
    <filename>{image_filename}</filename>
    <size>
        <width>{image_width}</width>
        <height>{image_height}</height>
        <depth>3</depth>
    </size>'''
    
    for label in labels:
        try:
            category = int(float(label[0]))
            xmin = int(float(label[1]) * image_width)
            xmax = int(float(label[2]) * image_width)
            ymin = int(float(label[3]) * image_height)
            ymax = int(float(label[4]) * image_height)
        except Exception:
            category = int(float(labels[0]))
            xmin = int(float(labels[1]) * image_width)
            xmax = int(float(labels[2]) * image_width)
            ymin = int(float(labels[3]) * image_height)
            ymax = int(float(labels[4]) * image_height)
        # if category == 0:
        #     object_name = 'dog'
        # elif category == 6:
        #     object_name = 'person'
        # else:
        #     object_name = 'unknown'
        object_name = category
        
        xml_content += f'''
    <object>
        <name>{object_name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>'''
    
    xml_content += '''
</annotation>'''
    
    return xml_content



def collect_data(train_split_ratio):
    # if os.path.isdir("data/train"):
    #     print("Folder with data already exists.")
    #     return
    
    print("Copying and splitting files from dataset.")
    source_img_directory = '/datasets/tdt4265/ad/NAPLab-LiDAR/images/'  # Directory containing image files
    source_label_directory = '/datasets/tdt4265/ad/NAPLab-LiDAR/labels_yolo_v1.1/'  # Directory containing label files
    destination_directory = 'data/'  # Directory where split data will be saved

    val_ratio = (1 - train_split_ratio)/2
    split_data(source_img_directory, source_label_directory, destination_directory, train_ratio=train_split_ratio, val_ratio=val_ratio, test_ratio=val_ratio, random_seed=0)

def create_labels(label_dir):
    label_files = os.listdir(label_dir)


collect_data(train_split_ratio=0.7)