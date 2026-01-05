import json
import os
import pickle
import random
import statistics
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import tqdm
import yaml

# Defines the tranches and sequences, which should be used for the train, validation and test split
standard_kia_split = {
    'train_set': {
        'bit': {  # BIT Technologies
            'tranche3': [70, 71, 72, 73, 74, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                         96, 97, 98, 99, 100, 101, 102, 103, 105, 107, 124],
            'tranche4': [149, 150, 151, 152, 153, 154, 155, 156, 177, 178, 179, 180, 181, 182, 184, 185, 187, 192, 193,
                         194, 211, 212, 213, 214, 215, 216],
            '5': [263, 270]
        },
        'mv': {  # Mackevision
            'tranche4': [40, 41, 42, 43, 44, 45, 51],
            '5': [53, 56, 61],
            '6': [66, 67],
            '7': []
        }
    },
    'val_set': {
        'bit': {
            'tranche3': [77, 78, 104, 106, 108, 114, 116, 125],
            'tranche4': [147, 148, 171, 172, 173],
            '5': [250]
        },
        'mv': {
            'tranche4': [46],
            '5': [57],
            '6': [],
            '7': []
        }
    },
    'test_set': {
        'bit': {
            'tranche3': [109, 110, 111, 112, 115, 117, 118, 119, 120, 121, 122, 123, 126, 127],
            'tranche4': [174, 209],
            '5': [251, 252, 264, 265, 271, 272, 273, 301, 302, 303, 310, 311, 312, 320, 321, 322, 484]
        },
        'mv': {
            'tranche4': [47, 48, 50],
            '5': [52, 54, 55, 58, 59, 60, 62, 63, 64],
            '6': [65, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82],
            '7': [83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96]
        }
    }
}


def get_transform(train):
    """
    Specifies all preprocessing steps and returns an instance from the class Compose for applying them.

    :param train: Whether the preprocessing is being applied on the train dataset. In this case, it will include a
    random horizontal flip data augmentation step. Options: True or False.
    :return: An instance of the class Compose, which can be called to apply the preprocessing steps on an image sample
    and the target labels.
    """
    transforms = [ToTensor()]
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


class ToTensor(nn.Module):
    """
    Class for loading image samples into PyTorch tensors.
    """

    def forward(self, image, target):
        """
        Converts the image sample from PIL to PyTorch tensor.

        :param image: An image sample, given as a PIL Image instance.
        :param target: A dictionary containing the 2D detection labels for the given image sample.
        :return: An image sample converted to a PyTorch tensor, and the target labels (unchanged).
        """
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)  # Image tensor type -> torch.float32
        return image, target


class Compose:
    """
    Class for applying the preprocessing steps on the image sample and target labels.
    """

    def __init__(self, transforms):
        """
        :param transforms: List of preprocessing steps, that should be applied.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Once this method is called, it will iterate over all preprocessing steps specified inside the transforms
        attribute and apply them to the image sample and labels, which are given as arguments to this method.

        :param image: An image sample, given as a PIL Image instance.
        :param target: A dictionary containing the 2D detection labels for the given image sample.
        :return: The preprocessed image sample and target labels.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def load_kia_dataloaders(cfg):
    """
    Loads the train, validation and test dataloader instances for the KIA dataset, used during the training process.
    Note that only the train dataset may contain keypoint/instance-segmentation annotations. The evaluation process
    takes only the bounding box predictions into account, hence the validation and test dataloader instances only
    contain bounding box annotations.

    :param cfg: A configuration dictionary containing the following parameter values: 'seed' is an integer number
    that should be used as the seed for the random module in order to support reproducibility, 'model_name' is the
    string name of the detection model, 'data_path' contains the string path to the dataset root folder,
    'min_obj_pixels' specifies the visible pixel amount threshold for filtering pedestrians, 'max_occl' and
    'min_occl' specify the occlusion threshold for filtering pedestrians, 'max_distance' specifies the distance
    threshold for filtering pedestrians, 'batch_size' specifies the amount of image samples in a single training
    batch and 'num_workers' specifies the amount of sub-processes for loading the data.
    :return: A train, validation and test dataloader instance, containing samples from the KIA dataset.
    """
    # Sets the random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    # Checks whether instance segmentation or keypoint labels should be loaded
    instance_segm = 'IS' in cfg['model_name']
    keypoint_det = 'KD' in cfg['model_name']
    # Loads the KIA dataset instances for training, validation and testing of the detection model
    train_dataset = KIA_dataset(
        cfg['data_path'],
        get_transform(train=False),
        min_obj_pixels=cfg['min_obj_pixels'],
        max_occl=cfg['max_occl'],
        min_occl=cfg['min_occl'],
        max_distance=cfg['max_distance'],
        instance_segm=instance_segm,
        keypoint_detect=keypoint_det
    )
    val_dataset = KIA_dataset(cfg['data_path'])
    test_dataset = KIA_dataset(cfg['data_path'])

    # Used for storing the sample indices for the train, validation and test samples, based on the standard KIA split
    train_indices = []
    val_indices = []
    test_indices = []

    # Iterates over all dataset samples to determine whether they belong to the train, validation or test split
    for sample_idx, img_sample_path in enumerate(
            train_dataset.imgs):  # f'{tranche}/{sequence}/'sensor/camera/left/png'/{file_name}.png'
        sample_info = img_sample_path.split('/')
        company_name = sample_info[0].split('_')[2]
        tranche = sample_info[0].split('_')[-1]
        sequence = int(sample_info[1].split('_')[3].split('-')[0].lstrip(
            '0'))  # A set of characters to remove as leading characters
        if sequence in standard_kia_split['train_set'][company_name][tranche]:
            train_indices.append(sample_idx)
        elif sequence in standard_kia_split['val_set'][company_name][tranche]:
            val_indices.append(sample_idx)
        elif sequence in standard_kia_split['test_set'][company_name][tranche]:
            test_indices.append(sample_idx)

    if keypoint_det:
        # Filters the list of train indices w.r. to the samples that contain keypoint annotations
        train_indices = train_dataset.get_keypoint_indices(train_indices)

    # Splits the dataset samples into train, validation and test samples
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    # Loads the train, validation and test dataset into PyTorch dataloader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,  # For the hook experiments, turn the shuffle off
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader


class KIA_dataset(torch.utils.data.Dataset):
    """
    Class used for loading the synthetic dataset from the KIA project. The data is used for training 2D detection models
    and supports also detection with instance segmentation or keypoint detection.
    """

    def __init__(self, root_path, transforms=get_transform(False), min_obj_pixels=None, max_occl=None,
                 min_occl=None, max_distance=None,
                 instance_segm=False, keypoint_detect=False):
        """
        Instantiates the class for handling the synthetic 2D detection dataset from the KIA project. Note: If the
        dataset instance is being used for evaluation (val or test split), then only the root_path needs to be set.

        :param root_path: The string path to the root folder of the synthetic dataset. The folder must contain all the
        tranches. Must be of type string.
        :param transforms: Transforms function, used for preprocessing the sample images and labels. Must be a function
        that takes two arguments. The PIL sample image and the target dictionary containing the label information.
        :param min_obj_pixels: Filter out all pedestrian instances (during training) that have a lower amount of visible
        instance pixels than here specified. Must be a number greater or equal to zero. Used to filter out small or
        heavily occluded pedestrians with a too low amount of visible instance pixels. Set this to 1 to include all
        visible pedestrian instances during training. Set this to None if the data is being loaded for evaluation.
        :param max_occl: Filter out all pedestrian instances (during training) that have a higher occlusion rate than
        here specified. Must be a decimal number between 0 and 1.0. Higher values mean that samples with higher
        occlusions are included during training. Set this to 0.99 to include all visible pedestrian instances during
        training. Set this to None if the data is being loaded for evaluation.
        :param min_occl: Filter out all pedestrian instances (during training) that have a lower occlusion rate than
        here specified. Must be a decimal number between 0 and 1.0.
        :param max_distance: Filter out all pedestrian instances (during training) that have a higher distance than here
        specified. Must be a number greater or equal to zero, that specifies the distance in meters. Set this to 100 to
        include all pedestrian instances during training (100 meters is the highest annotated distance value). Set this
        to None if the data is being loaded for evaluation.
        :param instance_segm: Whether to load the instance segmentation labels. Must be boolean value. Set this only
        for training.
        :param keypoint_detect: Whether to load the pedestrian keypoint labels. Must be boolean value. Set this only
        for training.
        """
        self.root_path = root_path
        self.transforms = transforms
        self.min_obj_pixels = min_obj_pixels
        self.max_occl = max_occl
        self.min_occl = min_occl
        self.max_distance = max_distance
        self.instance_segm = instance_segm
        self.keypoint_detect = keypoint_detect

        self.classes = {'human': 1}  # Maps the class names with their id (id 0 is reserved for the background class)
        self.amount_keypoints = 19  # Amount of annotated pedestrian keypoints within the KIA dataset
        # Defines the keypoint names that are being tracked
        # (tranche 6 & 7 introduced new keypoints that are skipped out of consistency reasons)
        self.keypoint_names = [
            'neck', 'shoulder_r', 'elbow_r', 'wrist_r', 'shoulder_l', 'elbow_l', 'wrist_l', 'pelvis', 'hip_r', 'knee_r',
            'heel_r', 'hip_l', 'knee_l', 'heel_l', 'eye_r', 'eye_l', 'toe_r', 'toe_l', 'shoulder_avg'
        ]
        # Defines a global list of keypoint indices for the flipping order
        # (e.g. index of eye_r replaces the index of eye_l etc.)
        global flip_keypoint_inds
        flip_keypoint_inds = [0, 4, 5, 6, 1, 2, 3, 7, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16, 18]

        # KIA dataset path postfixes
        self.img_sample_postfix = 'sensor/camera/left/png'
        self.bbox_2D_postfix = 'ground-truth/2d-bounding-box_json'
        self.instance_segm_postfix = 'ground-truth/semantic-instance-segmentation_png'
        self.keypoint_2D_postfix = 'ground-truth/2d-skeletons_json'
        self.depth_postfix = 'ground-truth/depth_exr'
        self.general_frame_postfix = 'ground-truth/general-globally-per-frame-analysis_json'

        # Stores the relative paths (within the KIA dataset), to all (required) dataset files
        self.imgs, self.bbox_jsons, self.instance_segm_imgs, self.keypoint_jsons = self.get_file_paths()

    def __len__(self):
        """
        :return: Amount of image samples.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Queries the KIA data at runtime, by indexing the precomputed sample list holding the relative paths to all image
        samples. The image and the corresponding labels are loaded and preprocessed by a transforms function, before
        being returned.

        :param idx: An integer value, used for indexing the KIA dataset. Must be an integer number between 0 and
        (sample amount - 1).
        :return: Two elements are being returned. The first one is the image sample, which is loaded by PIL and
        preprocessed by a transforms function. The second element is a target dictionary, holding the following key
        value pairs: boxes (key) -> A float tensor of size [N, 4] holding the bbox coordinates for each instance,
        labels (key) -> An integer tensor of size [N] holding the class id's for each instance, area (key) -> A float
        tensor of size [N] holding the bbox area for each instance, ignore (key) -> An integer tensor of size [N]
        holding the information whether the pedestrian instance is safety relevant (0) or should be ignored (1),
        instances_id(key) -> An integer tensor of size [N] holding the instance id's for each instance,
        No_bboxes_matched(key) -> A bool indicating if the input image containing the occluded pedestrians meeting
        the occlusion ratio threshold, image_path(key) -> A string holding the input image's path,
        occlusion_ratio_est(key) -> An float tensor of size [N] holding the estimated occlusion ratio of each instance,
        masks (key) -> An integer tensor of size [N, H, W] holding the binary instance segmentation masks for each
        instance, keypoints (key) -> A float tensor of size [N, K, 3] (K is the amount of annotated keypoints) holding
        the keypoint coordinates and visibility information for each keypoint.
        """
        # Path to the image sample and 2D bbox annotation json file
        img_path = f'{self.root_path}/{self.imgs[idx]}'
        bbox_json_path = f'{self.root_path}/{self.bbox_jsons[idx]}'
        img = Image.open(img_path).convert('RGB')  # Loads the image sample as a PIL Image

        # Extracts the 2D bbox information from the annotation json file -> this is a single json file for this pedestrian
        instances, bboxes, class_ids, bbox_areas, ignore, occlusion_ratio = self.extract_2D_bbox_data(bbox_json_path,
                                                                                                      img)

        if len(instances) != 0:
            # Stores all the 2D bbox information inside a target dictionary
            target = {
                'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
                'labels': torch.as_tensor(class_ids, dtype=torch.int64),
                'area': torch.as_tensor(bbox_areas, dtype=torch.float32),
                'ignore': torch.as_tensor(ignore, dtype=torch.uint8),
                'instances_id': torch.as_tensor(instances, dtype=torch.int64),
                'No_bboxes_matched': False,
                'image_path': img_path,
                'occlusion_ratio_est': torch.as_tensor(occlusion_ratio, dtype=torch.float32)
            }
            if self.instance_segm:
                # Path to the instance segmentation image
                instance_img_path = f'{self.root_path}/{self.instance_segm_imgs[idx]}'
                instance_segm_masks = self.extract_instance_segm_masks(instance_img_path, instances)
                target['masks'] = torch.as_tensor(instance_segm_masks, dtype=torch.uint8)
            if self.keypoint_detect:
                if not self.instance_segm:
                    # Path to the instance segmentation image
                    instance_img_path = f'{self.root_path}/{self.instance_segm_imgs[idx]}'
                    instance_segm_masks = self.extract_instance_segm_masks(instance_img_path, instances)
                # Path to the keypoint annotation json file
                keypoint_json_path = f'{self.root_path}/{self.keypoint_jsons[idx]}'
                keypoints = self.extract_2D_keypoint_data(keypoint_json_path, instances, instance_segm_masks, img)
                target['keypoints'] = torch.as_tensor(keypoints, dtype=torch.float32)
        else:
            # Stores the target information for an image sample without any objects
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "ignore": torch.zeros(0, dtype=torch.uint8),
                'instances_id': torch.zeros(0, dtype=torch.int64),
                'No_bboxes_matched': True,
                'image_path': img_path,
                'occlusion_ratio_est': torch.zeros(0, dtype=torch.float32)
            }
            if self.instance_segm:
                target["masks"] = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            if self.keypoint_detect:
                target["keypoints"] = torch.zeros((0, self.amount_keypoints, 3), dtype=torch.float32)

        # Applies the preprocessing on the image sample and annotation dictionary, using the transforms function
        img, target = self.transforms(img, target)
        return img, target

    def get_file_paths(self):
        """
        Extracts the relative paths (within the KIA dataset) to the dataset files. Every (relative) path is made up from
        the following components: {tranche name}/{sequence name}/{postfix (depending on the type of data)}/{file name}.

        :return: Four lists containing the relative paths (within the KIA root folder) to every: image sample, bbox
        annotation json file, instance segmentation image and keypoint annotation json file (if it exists). Every sample
        must have a consistent index within all lists, so it can be queried accordingly.
        """
        # Every sample must have a consistent index within all lists
        img_file_paths = []  # Relative paths to the image samples
        bbox_json_file_paths = []  # Relative paths to the bounding box annotation json files
        instance_img_file_paths = []  # Relative paths to the instance segmentation images
        keypoint_json_file_paths = []  # Relative paths to the keypoint annotation json files

        tranches = sorted(os.listdir(f'{self.root_path}'))
        for tranche in tqdm.tqdm(tranches):  # Iterates over all tranches
            sequences = sorted(os.listdir(f'{self.root_path}/{tranche}'))
            for sequence in sequences:  # Iterates over all sequences
                file_names = sorted(os.listdir(f'{self.root_path}/{tranche}/{sequence}/{self.img_sample_postfix}'))
                for file in file_names:  # Iterates over all files
                    data_path = f'{tranche}/{sequence}'  # Relative path to a specific sequence within a tranche
                    file_name = file[:-4]  # Removes the .png at the end of the file name
                    # Defines the relative paths to the dataset files
                    img_sample_path = f'{data_path}/{self.img_sample_postfix}/{file_name}.png'
                    instance_segm_path = f'{data_path}/{self.instance_segm_postfix}/{file_name}.png'
                    bbox_2D_path = f'{data_path}/{self.bbox_2D_postfix}/{file_name}.json'
                    keypoint_2D_path = f'{data_path}/{self.keypoint_2D_postfix}/{file_name}.json'
                    depth_path = f'{data_path}/{self.depth_postfix}/{file_name}.exr'
                    # Skip samples that are missing crucial annotations
                    if not os.path.exists(f'{self.root_path}/{bbox_2D_path}') or \
                            not os.path.exists(f'{self.root_path}/{depth_path}') or \
                            not os.path.exists(f'{self.root_path}/{instance_segm_path}'):
                        continue

                    # Adds the file path information
                    img_file_paths.append(img_sample_path)
                    bbox_json_file_paths.append(bbox_2D_path)
                    instance_img_file_paths.append(instance_segm_path)
                    # Checks whether the keypoint labels exist
                    if os.path.exists(f'{self.root_path}/{keypoint_2D_path}'):
                        keypoint_json_file_paths.append(keypoint_2D_path)
                    else:
                        keypoint_json_file_paths.append(None)
        return img_file_paths, bbox_json_file_paths, instance_img_file_paths, keypoint_json_file_paths

    def get_keypoint_indices(self, indices):
        """
        Determines, which samples contain keypoint annotations, based on a given list of sample indices.

        :param indices: A list of sample indices, which should be filtered into a list of keypoint indices.
        :return: A list of sample indices, which contain keypoint annotations.
        """
        keypoint_indices = []
        for keypoint_idx in indices:
            if self.keypoint_jsons[keypoint_idx] is not None:
                keypoint_indices.append(keypoint_idx)
        return keypoint_indices

    def extract_2D_bbox_data(self, json_path, img):
        """
        Loads the 2D bbox annotation json file and extracts the pedestrian information from it.

        :param json_path: Path to the annotation json file.
        :param img: A PIL Image instance of the sample image, corresponding to the json annotation file.
        :return: A list of instance id's (string), a list of lists containing bounding box coordinates [x1, y1, x2, y2]
        as integer values, a list of class id's (integer), a list of bounding box areas (integer) and a binary list of
        zeros and ones representing whether the pedestrian instance can be ignored during evaluation (not safety
        relevant).
        """
        with open(json_path) as json_file:
            labels = json.load(json_file)  # Loads the annotations from the json file

        # Placeholder for storing the annotation information
        instances = []  # A list of instance id's
        bboxes = []
        class_ids = []
        bbox_areas = []
        ignore = []
        occl_ratio = []
        # Iterates over all annotated instances from the bbox annotation json file
        for obj in labels:
            # Filters out objects that are not of the desired pedestrian class
            if labels[obj]['class_id'] == 'human':
                # Stores the bounding box information (converted to the corner format and clamped) (x,y,w,h) -> (x1,y1,x2,y2)
                x1 = min(img.width - 1, max(0, labels[obj]['c_x'] - int(labels[obj]['w'] / 2)))
                y1 = min(img.height - 1, max(0, labels[obj]['c_y'] - int(labels[obj]['h'] / 2)))
                x2 = min(img.width - 1, max(0, labels[obj]['c_x'] + int(labels[obj]['w'] / 2)))
                y2 = min(img.height - 1, max(0, labels[obj]['c_y'] + int(labels[obj]['h'] / 2)))
                if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                    # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                    continue
                if labels[obj]['instance_pixels'] < 1 or labels[obj]['occlusion_est'] > 0.99:
                    # Skips the instance if it is not visible at all
                    continue
                # Checks whether the data is being loaded for training or for evaluation (everything set to None)
                if self.min_obj_pixels is not None and self.max_occl is not None and self.max_distance is not None:
                    # Filters out objects that are too occluded, too small or too far away (only for training)
                    if (labels[obj]['instance_pixels'] < self.min_obj_pixels or
                            labels[obj]['occlusion_est'] < self.min_occl or
                            labels[obj]['occlusion_est'] > self.max_occl or
                            min(labels[obj]['distance'], 100) > self.max_distance):
                        continue
                # print(labels[obj]['occlusion_est'])
                instances.append(int(obj))  # Stores the instance id
                bboxes.append([x1, y1, x2, y2])  # Stores the bbox coordinates
                class_ids.append(self.classes[labels[obj]['class_id']])  # Stores the class information
                bbox_areas.append((x2 - x1) * (y2 - y1))  # Stores the bounding box area information
                ignore.append(int(labels[obj]['distance'] > 50))  # Ignore instances with a distance > 50m
                occl_ratio.append(labels[obj]['occlusion_est'])

        return instances, bboxes, class_ids, bbox_areas, ignore, occl_ratio

    def extract_instance_segm_masks(self, img_path, instances):
        """
        Loads the instance segmentation image and extracts the binary instance segmentation masks.

        :param img_path: Path to the instance segmentation image.
        :param instances: A list of instance id's (strings), generated by the extract_2D_bbox_data function.
        :return: A numpy list of binary instance segmentation masks in numpy format.
        """
        # Loads the instance segmentation image and converts it to RGB
        instance_segm_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        instance_masks = []  # Used for storing the binary instance segmentation masks for each bbox instance
        # Iterates over all bbox instances to produce the corresponding instance segmentation mask
        for obj_id in instances:
            # Converts the instance id into a hexadecimal color code
            hex_color_encoding = hex(int(obj_id)).replace('0x', '').zfill(6)
            # Converts the hexadecimal color code into an RGB color code
            rgb_color = np.array([int(hex_color_encoding[i:i + 2], 16) for i in (0, 2, 4)])
            # Produces a binary mask for the given RGB color code
            mask = cv2.inRange(instance_segm_img, rgb_color, rgb_color) / 255
            instance_masks.append(mask)  # Stores the produced mask for the current instance
        return np.array(instance_masks)

    def extract_2D_keypoint_data(self, json_path, instances, instance_segm_masks, img):
        """
        Loads the 2D keypoint annotation json file and extracts the information from it. Uses the instance segmentation
        masks to determine which keypoints are occluded.

        :param json_path: Path to the annotation json file.
        :param instances: A list of instance id's (strings), generated by the extract_2D_bbox_data function.
        :param instance_segm_masks: A numpy list of binary instance segmentation masks in numpy format.
        :param img: A PIL Image instance of the sample image, corresponding to the json annotation file.
        :return: A list of dimension [N, 19, 3], where N is the amount of instances, 19 is the amount of annotated
        keypoints (for each instance) and 3 is the amount of annotations belonging to every keypoint. Every keypoint
        has an x and y integer coordinate and a visibility score (0 or 1), which is determined based on the instance's
        segmentation mask.
        """
        with open(json_path) as json_file:
            labels = json.load(json_file)  # Loads the annotations from the json file

        keypoints = []  # Placeholder for storing the keypoint annotation information for all instances (N elements)
        for obj_idx, obj_id in enumerate(instances):
            obj_keypoints = []  # Stores all keypoints for a single instance (19 elements)
            for keypoint in self.keypoint_names:
                # Tranche 6 & 7 introduced missing keypoints to signalize their visibility
                if keypoint in labels[obj_id]:
                    # Image coordinates of a specific keypoint
                    x = labels[obj_id][keypoint][0]
                    y = labels[obj_id][keypoint][1]
                    # Checks whether the keypoint coordinates are inside the sample image
                    if 0 <= x < img.width and 0 <= y < img.height:
                        # Checks whether the keypoint coordinates lie inside the instance segmentation mask
                        if instance_segm_masks[obj_idx][y, x]:
                            # Stores the x and y keypoint coordinates and sets the visibility to 1
                            obj_keypoints.append([x, y, 1])  # (3 elements)
                            continue
                # Sets the x and y coordinates to 0, because the keypoint is not visible
                obj_keypoints.append([0, 0, 0])
            keypoints.append(obj_keypoints)
        return keypoints

    def preprocess_raw_KIA_LaG(self):
        """
        This function should be called right after the raw KIA data has been downloaded to process it for the first
        time. The  function loads the depth maps for each sample image to extract the distance information for each
        pedestrian instance. The distance information is finally stored within te original 2D bbox json annotation files
        (2d-bounding-box_json). Furthermore, the information about the amount of visible instance pixels and occlusion
        ratio has been refactored into the general-globally-per-frame-analysis_json files, for tranches 6 & 7. This
        information is extracted and written back to the original 2d-bounding-box_json files, in order to be consistent
        with the other tranches. However, it can occur that some of these samples are missing this information inside
        the general-globally-per-frame-analysis_json file. In this case, the information about the amount of visible
        instance pixels is extracted from the instance segmentation mask by counting. For estimating the occlusion
        ratio, a regression model is being used that is trained on the fully annotated samples to predict the occlusion
        ratio based on the bounding box area, the amount of visible instance pixels, the ratio of empty bounding box
        rows and the ratio of empty bounding box columns (compared to the total amount of bounding box rows and
        columns).
        """
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

        # Used for collecting data to train a regression model for occlusion estimation based on the bounding box area,
        # the amount of visible instance pixels and the ratio of empty bounding box rows & columns
        # (used to replace missing occlusion annotations)
        regressor_train = []  # Collects training data
        regressor_labels = []  # Collects labels (annotated occlusion_est values)

        tranches = sorted(os.listdir(f'{self.root_path}'))
        # Iterates over all tranches
        for tranche in tqdm.tqdm(tranches, desc='Loading Tranches', position=0):
            sequences = sorted(os.listdir(f'{self.root_path}/{tranche}'))
            # Iterates over all sequences
            for sequence in tqdm.tqdm(sequences, desc='Loading Sequences', position=1, leave=False, postfix=tranche):
                file_names = sorted(os.listdir(f'{self.root_path}/{tranche}/{sequence}/{self.img_sample_postfix}'))
                # Iterates over all files
                for file in tqdm.tqdm(file_names, desc='Loading Files', position=2, leave=False, postfix=sequence):
                    data_path = f'{self.root_path}/{tranche}/{sequence}'  # Path to a specific sequence within a tranche
                    file_name = file[:-4]  # Removes the '.png' postfix
                    raw_file_name = '-'.join(file_name.split('-')[2:])  # Removes the 'arb-camera001-' prefix

                    # Defines the paths to the dataset sample files
                    instance_segm_path = f'{data_path}/{self.instance_segm_postfix}/{file_name}.png'
                    bbox_2D_json_path = f'{data_path}/{self.bbox_2D_postfix}/{file_name}.json'
                    depth_map_path = f'{data_path}/{self.depth_postfix}/{file_name}.exr'
                    general_frame_json_path = f'{data_path}/{self.general_frame_postfix}/world-{raw_file_name}.json'

                    # Skip samples that are missing crucial annotations (these samples will not be used anymore)
                    if not os.path.exists(instance_segm_path) or \
                            not os.path.exists(bbox_2D_json_path) or \
                            not os.path.exists(depth_map_path):
                        continue

                    # Checks, which information is missing from the bbox 2D json files
                    instances = {}
                    with open(bbox_2D_json_path) as json_file:
                        bbox_labels = json.load(json_file)  # Loads the annotations from the json file
                    # Iterate over all annotated objects
                    for obj in bbox_labels:
                        if bbox_labels[obj]['class_id'] == 'human':
                            # Stores the bounding box information (converted to the corner format and clamped)
                            x1 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] - int(bbox_labels[obj]['w'] / 2)))
                            y1 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] - int(bbox_labels[obj]['h'] / 2)))
                            x2 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] + int(bbox_labels[obj]['w'] / 2)))
                            y2 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] + int(bbox_labels[obj]['h'] / 2)))
                            # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                            if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                                continue
                            # Tracks the presence of required annotations
                            instance_pixels, occlusion_est, distance = False, False, False
                            if 'instance_pixels' in bbox_labels[obj]:
                                instance_pixels = True
                            if 'occlusion_est' in bbox_labels[obj]:
                                occlusion_est = True
                            if 'distance' in bbox_labels[obj]:
                                distance = True
                            instances[obj] = {
                                'instance_pixels': instance_pixels,
                                'occlusion_est': occlusion_est,
                                'distance': distance
                            }

                    if instances != {}:  # Skip samples that do not contain any objects
                        update_annotations = False  # Tracks whether the annotations need to be updated or not
                        # Extracts the instance segmentation mask for each pedestrian instance
                        instance_segm_masks = self.extract_instance_segm_masks(instance_segm_path, instances)

                        # Extracts the distance information, if it is missing
                        if not distance:
                            update_annotations = True
                            # Loads the depth map
                            depth_img = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED)
                            if len(depth_img.shape) != 2:
                                # Tranches 6 & 7 have multichannel depth maps (take only the first channel)
                                depth_img = depth_img[:, :, 0]
                            # Iterates over all pedestrian instances
                            for obj_order, obj in enumerate(instances):
                                # Masks out the depth map, using the instance segmentation mask
                                masked_depth = depth_img.copy() * instance_segm_masks[obj_order].copy()
                                # Calculates the distance from the camera to the nearest point of the pedestrian
                                if np.any(masked_depth):
                                    distance = min(round(np.min(masked_depth[np.nonzero(masked_depth)]), 2), 100.0)
                                else:
                                    distance = 100.0
                                instances[obj]['distance'] = distance  # Stores the computed distance

                                # Extracts the information about the amount of visible instance pixels, if it is missing
                                if not instance_pixels:
                                    # Computes the amount of visible instance pixels, based on the segmentation
                                    visible_instance_pixels = np.count_nonzero(instance_segm_masks[obj_order])
                                    instances[obj]['instance_pixels'] = visible_instance_pixels

                        # Extracts the occlusion information, if it is missing
                        if not occlusion_est:
                            update_annotations = True
                            with open(general_frame_json_path) as json_file:
                                frame_labels = json.load(json_file)  # Loads the annotations from the json file
                            for entity in frame_labels['base_context']['entities']:
                                # Finds the corresponding annotations for each pedestrian instance
                                if str(entity['instance_id']) in instances:
                                    try:
                                        # Extracts the annotated amount of visible instance pixels (if it exists)
                                        instance_pixels = \
                                            entity['sensor_occlusion'][0]['occlusion']['visible_pixels']
                                        instances[str(entity['instance_id'])]['instance_pixels'] = instance_pixels
                                    except KeyError:
                                        pass  # Use the annotation from the previous step (segmentation based)

                                    try:
                                        # Extracts the occlusion information if it exists
                                        occlusion_est = round(entity['sensor_occlusion'][0]['occlusion']['rate'], 2)
                                        instances[str(entity['instance_id'])]['occlusion_est'] = occlusion_est
                                    except KeyError:
                                        pass  # No annotations -> need to approximate this in the next iteration

                        # Loads the bbox 2D json files to extract data for approximating the occlusion ratio
                        # Overwrites the distance, instance_pixels and occlusion_est if it is missing
                        with open(bbox_2D_json_path, 'r+') as json_file:
                            bbox_labels = json.load(json_file)  # Loads the annotations from the json file
                            for obj_idx, obj in enumerate(instances):
                                # Extracts the data for approximating the occlusion ratio
                                # Extracts clamped bbox coordinates
                                x1 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] - int(bbox_labels[obj]['w'] / 2)))
                                y1 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] - int(bbox_labels[obj]['h'] / 2)))
                                x2 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] + int(bbox_labels[obj]['w'] / 2)))
                                y2 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] + int(bbox_labels[obj]['h'] / 2)))
                                bbox_area = int((x2 - x1) * (y2 - y1))
                                bbox_instance_mask = instance_segm_masks[obj_idx][y1:y2, x1:x2]
                                empty_rows = sum(~np.any(bbox_instance_mask, axis=1))
                                empty_columns = sum(~np.any(bbox_instance_mask, axis=0))
                                empty_rows_ratio = round(empty_rows / (y2 - y1), 2)
                                empty_columns_ratio = round(empty_columns / (x2 - x1), 2)

                                # Overwrites the missing annotations
                                if 'distance' not in bbox_labels[obj]:
                                    bbox_labels[obj]['distance'] = instances[obj]['distance']
                                if 'instance_pixels' not in bbox_labels[obj]:
                                    bbox_labels[obj]['instance_pixels'] = instances[obj]['instance_pixels']
                                if 'occlusion_est' not in bbox_labels[obj]:
                                    # If the value is of type bool, it means that the occlusion information is missing
                                    # Otherwise, the value is numerical and can be overwritten
                                    if type(instances[obj]['occlusion_est']) != bool:
                                        bbox_labels[obj]['occlusion_est'] = instances[obj]['occlusion_est']
                                    else:
                                        # Store the information for approximating the occlusion_est
                                        bbox_labels[obj]['empty_rows_ratio'] = empty_rows_ratio
                                        bbox_labels[obj]['empty_columns_ratio'] = empty_columns_ratio

                                # If the occlusion_est was overwritten or if it was already annotated, it means that the
                                # sample is now fully annotated and eligible to be a training sample for the regressor
                                if 'occlusion_est' in bbox_labels[obj]:
                                    occlusion_est = bbox_labels[obj]['occlusion_est']
                                    # Store the data for training the regression model
                                    regressor_train.append(
                                        [
                                            bbox_area,
                                            bbox_labels[obj]['instance_pixels'],
                                            empty_rows_ratio,
                                            empty_columns_ratio
                                        ]
                                    )
                                    regressor_labels.append(occlusion_est)  # Labels (to be predicted later)

                                if update_annotations:
                                    # Overwrites the 2D bbox annotations inside the json file
                                    json_file.seek(0)
                                    json_file.write(json.dumps(bbox_labels, indent=4))
                                    json_file.truncate()

        # Splits the regressor data into train and test samples
        x_train, x_test, y_train, y_test = \
            train_test_split(np.array(regressor_train), np.array(regressor_labels), test_size=0.2, random_state=0)
        model = LinearRegression().fit(x_train, y_train)  # Trains the regression model, based on the aggregated data
        # Stores the regressor data and the trained regression model
        with open('occlusion_est_model.p', 'wb') as pickle_file:
            pickle.dump(
                {
                    'train_data': x_train,
                    'train_labels': y_train,
                    'test_data': x_test,
                    'test_labels': y_test,
                    'model': model
                },
                pickle_file,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        # Iterates a second time over all files to add the missing occlusion_est annotations using the approx. model
        tranches = sorted(os.listdir(f'{self.root_path}'))
        # Iterates over all tranches
        for tranche in tqdm.tqdm(tranches, desc='Loading Tranches', position=0):
            sequences = sorted(os.listdir(f'{self.root_path}/{tranche}'))
            # Iterates over all sequences
            for sequence in tqdm.tqdm(sequences, desc='Loading Sequences', position=1, leave=False, postfix=tranche):
                file_names = sorted(os.listdir(f'{self.root_path}/{tranche}/{sequence}/{self.img_sample_postfix}'))
                # Iterates over all files
                for file in tqdm.tqdm(file_names, desc='Loading Files', position=2, leave=False, postfix=sequence):
                    data_path = f'{self.root_path}/{tranche}/{sequence}'  # Path to a specific sequence within a tranche
                    file_name = file[:-4]  # Removes the '.png' postfix

                    # Defines the paths to the dataset sample files
                    instance_segm_path = f'{data_path}/{self.instance_segm_postfix}/{file_name}.png'
                    bbox_2D_json_path = f'{data_path}/{self.bbox_2D_postfix}/{file_name}.json'
                    depth_map_path = f'{data_path}/{self.depth_postfix}/{file_name}.exr'

                    # Skip samples that are missing crucial annotations (these samples will not be used anymore)
                    if not os.path.exists(instance_segm_path) or \
                            not os.path.exists(bbox_2D_json_path) or \
                            not os.path.exists(depth_map_path):
                        continue

                    # Approximate the occlusion information if it is missing
                    update = False  # Used to control, whether to update the json file
                    with open(bbox_2D_json_path, 'r+') as json_file:
                        bbox_labels = json.load(json_file)  # Loads the annotations from the json file
                        for obj in bbox_labels:
                            if bbox_labels[obj]['class_id'] == 'human':
                                # Locate pedestrian instances that are missing the occlusion information
                                if 'occlusion_est' not in bbox_labels[obj]:
                                    # Extracts the bbox parameters to approximate the occlusion ratio
                                    x1 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] - int(bbox_labels[obj]['w'] / 2)))
                                    y1 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] - int(bbox_labels[obj]['h'] / 2)))
                                    x2 = min(1920 - 1, max(0, bbox_labels[obj]['c_x'] + int(bbox_labels[obj]['w'] / 2)))
                                    y2 = min(1280 - 1, max(0, bbox_labels[obj]['c_y'] + int(bbox_labels[obj]['h'] / 2)))
                                    # Skips faulty bboxes
                                    if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                                        continue
                                    bbox_area = int((x2 - x1) * (y2 - y1))
                                    instance_pixels = bbox_labels[obj]['instance_pixels']
                                    empty_rows_ratio = bbox_labels[obj]['empty_rows_ratio']
                                    empty_columns_ratio = bbox_labels[obj]['empty_columns_ratio']
                                    del bbox_labels[obj]['empty_rows_ratio']
                                    del bbox_labels[obj]['empty_columns_ratio']
                                    # Mark the annotation as an approximation
                                    bbox_labels[obj]['occlusion_approx'] = True
                                    estimation_params = np.array(
                                        [
                                            bbox_area,
                                            instance_pixels,
                                            empty_rows_ratio,
                                            empty_columns_ratio
                                        ]
                                    )
                                    # Approximate the occlusion_est
                                    bbox_labels[obj]['occlusion_est'] = \
                                        round(np.clip(model.predict([estimation_params]).item(), 0.0, 1.0), 2)
                                    update = True

                        if update:
                            # Overwrites the 2D bbox annotations inside the json file
                            json_file.seek(0)
                            json_file.write(json.dumps(bbox_labels, indent=4))
                            json_file.truncate()


def collate_fn(batch):
    """
    Creates the batch of data samples and targets for the dataloader.

    :param batch: A list of tuples, where each tuple contains the image sample and the corresponding targets.
    :return: A tuple of two elements, where the first element contains a tuple of all image samples and the second
    element contains a tuple of the corresponding target labels.
    """
    return tuple(zip(*batch))


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Class for applying random horizontal image flipping to an image sample and the target labels.
    """

    def forward(self, image, target):
        """
        Applies randomly horizontal flipping to the image sample and adapts the target labels to be consistent with the
        flipped image.

        :param image: An image sample, given as a PyTorch tensor instance.
        :param target: A dictionary containing the 2D detection labels for the given image sample.
        :return: The image sample and target labels with either horizontal flipping applied to them or no change at all.
        """
        # Draws a random number to decide whether to apply the horizontal flip
        if torch.rand(1) < self.p:
            image = F.hflip(image)  # Applies horizontal flipping to the image sample
            if target is not None:
                width, _ = F.get_image_size(image)
                # Applies horizontal flipping to the bbox coordinates
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
                if "masks" in target:
                    # Applies horizontal flipping to the instance segmentation masks
                    target["masks"] = target["masks"].flip(-1)
                if "keypoints" in target:
                    # Applies horizontal flipping to the instance keypoints
                    keypoints = target["keypoints"]
                    keypoints = self._flip_person_keypoints(keypoints, width)
                    target["keypoints"] = keypoints
        return image, target

    def _flip_person_keypoints(self, kps, width):
        """
        Converts the person keypoint labels for the case that the given image sample is being horizontally flipped. Note
        that a global flipped_keypoint_inds variable must be defined, which contains a list of the keypoint indices
        for the horizontal flipping case (e.g. if the keypoints being tracked are
        ['eye_l', 'eye_r', 'shoulder_avg', 'arm_l', 'arm_r'],
        then in the case of horizontal flipping, the keypoints that are assigned to a left or a right side must be
        flipped too hence the flipped_keypoint_inds list would be defined as [1, 0, 2, 4, 3]).

        :param kps: A list of instances within a given sample image, where each instance consists of several keypoints
        that have x, y and visibility features annotated. The overall shape of the list is [N, K, 3], where N is the
        amount of instances, K is the amount of keypoints and 3 is the amount of annotated features per keypoint.
        :param width: The width of the image sample.
        :return: The converted kps list with the keypoint labels being consistent with the horizontally flipped sample
        image.
        """
        # Applies horizontal flipping to the keypoint coordinates
        # Note that flip_keypoint_inds is a global variable that defines the flipping order
        flipped_data = kps[:, flip_keypoint_inds]
        flipped_data[..., 0] = width - flipped_data[..., 0]
        # Determines which keypoints have visibility set to 0, and sets their coordinates also to 0 (convention)
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0
        return flipped_data


def load_trained_faster_rcnn(weight_path_faster_rcnn):
    """
    load the pre-trained Faster RCNN detector with backbone ResNet50.

    :param weight_path_faster_rcnn:
    :return:
    """
    # device info
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)  # Sets the GPU, which should be used for training/evaluation
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device used for training/evaluation: {device}')

    # load the model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained_backbone=True,  # a model with backbone pre-trained on Imagenet
        trainable_backbone_layers=5,  # all backbone layers are trainable
        num_classes=2  # number of output classes of the model (including the background)
    )

    checkpoint = torch.load(weight_path_faster_rcnn)
    model.load_state_dict(checkpoint['model'])
    print('Pre-trained weights have been loaded!')

    return model, device


def dts_gts_match(batch_outputs, targets, score_thresh, iou_thresh=0.5):
    """
    Find the predicted bboxes which have the largest iou with the ground truth bboxes of the
    pedestrians, and finally return these matched predictions.
    :param batch_outputs: the predictions of one batch
    :param targets: ground truth of the corresponding batch
    :param score_thresh: threshold for filter out the predictions with a low confidence score
    :return: predictions of the pedestrians in the input image
    """
    # Iterates over all items within this batch
    for batch_idx in range(len(targets)):
        # Extracts the ground truth information for each object
        gts = {}
        for j in range(len(targets[batch_idx]['boxes'])):
            gts[j] = {
                'bbox': targets[batch_idx]['boxes'][j],
                'area': targets[batch_idx]['area'][j],
                'ignore': targets[batch_idx]['ignore'][j],
                'instances_id': targets[batch_idx]['instances_id'][j]
            }

        # Extracts the predictions for each detected object
        dts = {}
        for j in range(len(batch_outputs[batch_idx]['boxes'])):
            dts[j] = {
                'bbox': batch_outputs[batch_idx]['boxes'][j],
                'score': batch_outputs[batch_idx]['scores'][j]
            }

        # A matrix that stores the IoU value between each ground truth and predicted bbox
        gts_dts_ious = np.zeros((len(gts), len(dts)))

        # Checks whether there are any ground truths or predictions
        if len(gts) != 0 and len(dts) != 0:
            # Computes the IoU between each ground truth and each predicted bbox
            for gt_id in gts:
                for dt_id in dts:
                    # Computes the corner coordinates for the intersection
                    x1 = max(dts[dt_id]['bbox'][0], gts[gt_id]['bbox'][0])
                    y1 = max(dts[dt_id]['bbox'][1], gts[gt_id]['bbox'][1])
                    x2 = min(dts[dt_id]['bbox'][2], gts[gt_id]['bbox'][2])
                    y2 = min(dts[dt_id]['bbox'][3], gts[gt_id]['bbox'][3])
                    # Calculates the area for the intersection
                    inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
                    if inter_area == 0:
                        # No intersection -> IoU is 0
                        gts_dts_ious[gt_id][dt_id] = 0
                        continue

                    # Computes the area of the predicted bbox
                    detection_area = \
                        (dts[dt_id]['bbox'][2] - dts[dt_id]['bbox'][0]) * \
                        (dts[dt_id]['bbox'][3] - dts[dt_id]['bbox'][1])
                    target_area = gts[gt_id]['area']
                    iou = inter_area / (detection_area + target_area - inter_area)  # Computes the IoU
                    gts_dts_ious[gt_id][dt_id] = iou.item()  # Stores the IoU
        # Creates a copy of the gts_dts_iou matrix in order to filter it w.r. to the score_thresh
        gts_dts_ious_conf_thr = gts_dts_ious.copy()
        for dt_id in dts:
            if dts[dt_id]['score'].item() < float(score_thresh):
                # Resets the IoU values for detections with a confidence score below the threshold
                gts_dts_ious_conf_thr[:, dt_id] = 0

        matched_dts_idx = []  # A list used for tracking the dt_ids of detection bboxes
        matched_instances_id = []  # A list used to store the instance id og the matched GT
        # Matches the detections to the ground truth objects
        while True:
            if not np.any(gts_dts_ious_conf_thr):
                # Exit the loop if the matrix is an empty sequence
                break

            # Matrix coordinates of a ground truth and prediction bbox with the currently highest IoU
            match = np.unravel_index(gts_dts_ious_conf_thr.argmax(), gts_dts_ious_conf_thr.shape)

            if gts_dts_ious_conf_thr[match[0], match[1]] < iou_thresh:
                # Exit the loop if the highest IoU value is lower than the lowest IoU threshold
                break

            # Sets all IoU values for the matched prediction and gt bbox to 0, since they have been matched (except
            # for gt bboxes with the iscrowd property set to 1)
            if 'iscrowd' in gts[match[0]]:
                if gts[match[0]]['iscrowd'] == 0:
                    gts_dts_ious_conf_thr[match[0], :] = 0
            else:
                gts_dts_ious_conf_thr[match[0], :] = 0
                matched_instances_id.append(gts[match[0]]['instances_id'])  # append the id of matched instance GT
            gts_dts_ious_conf_thr[:, match[1]] = 0
            # append the index of the matched prediction
            matched_dts_idx.append(match[1])

        # test if there is any matched prediction
        if len(matched_dts_idx) == 0:
            matched_dts = []
            # append an empty prediction
            matched_dts.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                'scores': torch.zeros(0),
            })

        else:
            matched_dts = [{k: v[matched_dts_idx] for k, v in t.items()} for t in batch_outputs]
            matched_dts[batch_idx]['instances_id'] = matched_instances_id  # add the ids of the matched detections
    return matched_dts


def main():
    # Reads the yaml file, which contains the parameters for loading the training dataset
    cfg_path = './config/extract_activation_patches.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # load the dataloader for KIA training set
    train_loader, _, _ = load_kia_dataloaders(cfg)

    # load the pre-trained model
    weight_path = './model_weights/final_model.pth'
    model, device = load_trained_faster_rcnn(weight_path)
    model.eval()  # set the model to eval mode

    # send model to devices
    model.to(device)
    # a list to store all the last feature maps for all the samples in the training dataloader
    feature_maps = []
    # store the width and heights of the extracted feature maps for calculating the averaged shape later on
    extracted_width, extracted_height = [], []

    # register the forward hook
    model.backbone.body.layer4[2].conv3.register_forward_hook(lambda module, input, output: feature_maps.append(output))
    # specify the cpu device
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    # creat an empty dataframe to save the extracted activations
    df_all = pd.DataFrame(columns=['image_path', 'transformed coords', 'extracted_feature_maps'])

    # process the input image in training dataset one by one
    for idx, (image, target) in enumerate(tqdm.tqdm(train_loader)):
        # no GT bboxes meet the requirements of the criteria choosing the GT, so skip
        if target[0]['No_bboxes_matched']:
            continue
        # send the input image to GPU
        image = [img.to(device) for img in image]
        with torch.no_grad():
            # get the width and height of the input image
            c_img, h_img, w_img = image[0].size()  # [C, H, W]
            # feed the input image to model
            outputs = model(image)
            # move the results to cpu
            outputs_cpu = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # keep the detections which have the largest iou with the zero-occluded pedestrian ground truth bboxes
            outputs_matched = dts_gts_match(outputs_cpu, target, score_thresh=0.2, iou_thresh=0.25)
            # test if there is any matched prediction
            if outputs_matched[0]['boxes'].nelement() == 0:
                feature_maps.pop()  # discard the feature maps of this input image due to no-matched detections
                continue  # move to the next input image
            # get the corresponding feature map regarding the particular input image
            feature_map = feature_maps.pop()[0].clone()
            # Alternatively pass the activation through the ReLU
            # feature_map = torch.nn.ReLU()(feature_maps[-1][0].clone())
            # get the width and height of the feature map
            c_f, h_f, w_f = feature_map.shape  # with shape [C, H, W]
            # transform factors between bboxes and feature map
            w_factor, h_factor = w_f / w_img, h_f / h_img
            # apply the transform factor to the prediction bboxes
            refactored_bboxes = outputs_matched[0]['boxes'].clone()
            refactored_bboxes[:, [0, 2]] *= w_factor
            refactored_bboxes[:, [1, 3]] *= h_factor

            # turn the float coords into the integer coords
            refactored_bboxes[:, [0, 1]] = torch.floor(refactored_bboxes[:, [0, 1]])  # x1, x2 are floored
            refactored_bboxes[:, [2, 3]] = torch.ceil(refactored_bboxes[:, [2, 3]])  # x2, y2 are ceiled

            # extract the activation within the areas of refactored bboxes
            extracted_feature_maps = []
            for refactored_bbox in refactored_bboxes.int():
                # extract the corresponding pieces of feature maps
                extracted_feature_map = feature_map[:, refactored_bbox[1]:min(refactored_bbox[3], h_f),
                                        refactored_bbox[0]:min(refactored_bbox[2], w_f)]  # shape [C, H_e, W_e]
                # append the width and height
                extracted_height.append(extracted_feature_map.shape[1])  # height
                extracted_width.append(extracted_feature_map.shape[2])  # weight
                # append the extracted activation
                extracted_feature_maps.append(extracted_feature_map)
            # get the list of the instance IDs within the input image
            key_id = [t.item() for t in outputs_matched[0]['instances_id']]
            # creat a dict to store the transformed bboxes coordinates in array with the instance ID as keys
            merged_coords = dict(zip(key_id, refactored_bboxes.int().cpu().numpy()))
            # creat a dict to store the extracted activation patches in list with the instance ID as keys
            merged_extractions = dict(zip(key_id, [x.cpu().numpy() for x in extracted_feature_maps]))
            # append the path of the input image, transformed bboxes coordinates and the extracted activation patches
            # to the final dataframe
            df = pd.DataFrame(
                [[target[0]['image_path'], merged_coords, merged_extractions]],
                columns=['image_path', 'transformed coords', 'extracted_feature_maps'],
                index=[idx]  # here the idx is the index of the input image in the training dataloader
            )
            df_all = df_all.append(df)

    # calculate the averaged width and height of the extracted activations
    avg_width_extraction = round(statistics.mean(extracted_width))
    avg_height_extraction = round(statistics.mean(extracted_height))
    print(f'the averaged width of the extracted activations is {avg_width_extraction}')
    print(f'the averaged height of the extracted activations is {avg_height_extraction}')
    print(f'the averaged aspect-ratio is {round(avg_width_extraction / avg_height_extraction, 2)}')

    # revert the cpu thread
    torch.set_num_threads(n_threads)
    # save the dataframe to pickle file
    df_all.to_pickle('extracted_activation.pkl')


if __name__ == '__main__':
    main()
