import os
import csv
import random
import argparse
import pickle

from tqdm import tqdm
from PIL import Image
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/path/to/data',
                        help='Path to the folder containing the unzipped GTSRB train and test set '
                             '(GTSRB_Final_Training_Images and GTSRB_Final_Test_Images).')
    parser.add_argument('--size', type=int, default=299, help='Size value used for resizing all images.')
    parser.add_argument('--val_proportion', type=float, default=0.15,
                        help='Proportion of train images, which should be used for the validation dataset.')
    parser.add_argument('--seed', type=int, default=1,
                        help='The random seed for sampling the train images, which are used for the validation set.')
    parser.add_argument('--destination_path', type=str, default='/path/to/dest_dir',
                        help='Destination path for storing the final pickle files.')
    args = parser.parse_args()

    # Path to the GTSRB train and test set images
    train_folder_path = f"{args.data_path}/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
    test_folder_path = f"{args.data_path}/GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"

    # Placeholders for storing the train, validation and test images and the corresponding labels
    train_imgs = []
    train_labels = []
    val_imgs = []
    val_labels = []
    test_imgs = []
    test_labels = []

    # Fills the train and validation list
    print('Loading train and validation set.')
    for class_folder in tqdm(os.listdir(train_folder_path), desc='Processed class folders'):
        # Loads the train ROI labels from the .csv file for the train images
        train_roi_dic = {}  # Key is the image file name and value is the ROI label
        with open(f"{train_folder_path}/{class_folder}/GT-{class_folder}.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            next(csv_reader, None)  # Skips the header row
            for row in csv_reader:
                # Fills the dictionary with the ROI values
                train_roi_dic[row[0]] = (int(row[3]), int(row[4]), int(row[5]), int(row[6]))

        class_image_files = os.listdir(f"{train_folder_path}/{class_folder}")  # Loads the file names of all images
        random.Random(args.seed).shuffle(class_image_files)  # Shuffles the images inside the class folder
        # Calculates the amount of validation samples for the current class (based on the amount of class samples)
        val_imgs_amount = round(len(class_image_files) * args.val_proportion)
        val_counter = 0  # Counter for the amount of processed validation samples

        for image_file in class_image_files:
            if image_file.endswith('.ppm'):
                img = Image.open(f"{train_folder_path}/{class_folder}/{image_file}")
                img = img.crop(train_roi_dic[image_file])  # Cuts out the ROI
                img = img.resize((args.size, args.size))  # Resizes the image

                # Fills the validation list
                if val_counter < val_imgs_amount:
                    val_imgs.append(np.array(img))
                    val_labels.append(int(class_folder.lstrip('0')) if class_folder.lstrip('0') != '' else 0)
                    val_counter += 1
                # Fills the train list
                else:
                    train_imgs.append(np.array(img))
                    train_labels.append(int(class_folder.lstrip('0')) if class_folder.lstrip('0') != '' else 0)

    # Converts the train and validation lists to numpy arrays
    train_imgs = np.array(train_imgs)
    train_labels = np.array(train_labels)
    val_imgs = np.array(val_imgs)
    val_labels = np.array(val_labels)
    print('Loading finished. \n')

    # Loads the test labels from the .csv file
    print('Loading test set.')
    test_labels_dic = {}  # Key is the image file name and value is the label and ROI values
    with open(f"{test_folder_path}/Test.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)  # Skips the header row
        for row in csv_reader:
            # Fills the dictionary with the class label and the ROI values
            test_labels_dic[row[7]] = [row[6], (int(row[2]), int(row[3]), int(row[4]), int(row[5]))]

    # Fills the test list
    for image_file in tqdm(os.listdir(test_folder_path), desc='Processed image samples'):
        if image_file.endswith('.ppm'):
            img = Image.open(f"{test_folder_path}/{image_file}")
            img = img.crop(test_labels_dic['Test/' + image_file.replace('.ppm', '.png')][1])  # Cuts out the ROI
            img = img.resize((args.size, args.size))  # Resizes the image
            test_imgs.append(np.array(img))
            test_labels.append(int(test_labels_dic['Test/' + image_file.replace('.ppm', '.png')][0]))

    # Converts the test lists to numpy arrays
    test_imgs = np.array(test_imgs)
    test_labels = np.array(test_labels)
    print('Loading finished. \n')

    # Makes a new folder for storing the images and labels inside pickle files
    if not os.path.exists(f"{args.destination_path}/pickled_data"):
        os.mkdir(f"{args.destination_path}/pickled_data")

    print('Generating .pickle files.')
    with open(f"{args.destination_path}/pickled_data/train.p", 'wb') as f:
        pickle.dump({'features': train_imgs, 'labels': train_labels}, f, protocol=4)  # Stores the train set
    print(f"train.p has been generated -> {args.destination_path}/pickled_data/train.p")

    with open(f"{args.destination_path}/pickled_data/valid.p", 'wb') as f:
        pickle.dump({'features': val_imgs, 'labels': val_labels}, f, protocol=4)  # Stores the validation set
    print(f"valid.p has been generated -> {args.destination_path}/pickled_data/valid.p")

    with open(f"{args.destination_path}/pickled_data/test.p", 'wb') as f:
        pickle.dump({'features': test_imgs, 'labels': test_labels}, f, protocol=4)  # Stores the test set
    print(f"test.p has been generated -> {args.destination_path}/pickled_data/test.p")
    print('All pickle files have been generated!')


if __name__ == '__main__':
    main()
