import itertools
import json
import os
import pickle
import random

import cv2
import numpy as np
from PIL import Image
from skimage.measure import shannon_entropy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
import tqdm
from collections import defaultdict

import utils

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


def load_dataloaders(cfg):
    """
    Loads the dataloaders for the selected dataset.

    :param cfg: A configuration dictionary containing the name of the selected dataset.
    :return: A train, validation and test dataloader instance, used for training and evaluating the detection model.
    """
    if cfg['dataset'] == 'KIA':
        train_loader, val_loader, test_loader = load_kia_dataloaders(cfg)
    elif cfg['dataset'] == 'CityPersons':
        train_loader, val_loader, test_loader = load_citypersons_dataloaders(cfg)
    elif cfg['dataset'] == 'EuroCityPersons':
        train_loader, val_loader, test_loader = load_eurocitypersons_dataloaders(cfg)

    return train_loader, val_loader, test_loader


def load_eurocitypersons_dataloaders(cfg):
    """
    Loads the train, validation and test dataloader instances for the EuroCity Persons dataset, used during the training
    process. Note that this dataset only include the bounding boxes annotation.

    Note that the original EuroCity Persons test dataset does not have public labels,
    therefor the val set is loaded as the test set.

    :param cfg: A configuration dictionary containing the following parameter values: 'seed' is an integer number
    that should be used as the seed for the random module in order to support reproducibility, 'model_name' is the
    string name of the detection model, 'data_path' contains the string path to the dataset root folder,
    'batch_size' specifies the amount of image samples in a single training batch and 'num_workers'
    specifies the amount of sub-processes for loading the data.
    :return: A train, validation and test dataloader instance, containing samples from the EuroCity Persons dataset.
    """
    # Sets the random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])

    # Loads the EuroCity Persons dataset instances for training, validation and testing of the detection model
    root = cfg['data_path']
    # path of the annotations, The annotations include 'pedestrian' and 'rider' sets
    train_ann = f'{root}/ECP/day/coco_format_labels/day_train_all.json'
    val_ann = f'{root}/ECP/day/coco_format_labels/day_val_all.json'

    train_dataset = EuroCityPersons_dataset(
        cfg['data_path'],
        train_ann,
        transforms=get_transform(train=True),
    )

    val_dataset = EuroCityPersons_dataset(
        cfg['data_path'],
        val_ann,
    )

    # due to the test-set don't have public annotations available
    test_dataset = val_dataset

    # Loads the train, validation and test dataset into PyTorch dataloader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    return train_loader, val_loader, test_loader


def load_citypersons_dataloaders(cfg):
    """
    Loads the train, validation and test dataloader instances for the CityPersons dataset, used during the training
    process. Note that only the train dataset may contain instance-segmentation annotations. The evaluation process
    takes only the bounding box predictions into account, hence the validation and test dataloader instances only
    contain bounding box annotations.
    
    Note that the original CityPersons test dataset does not have public labels,
    therefor the validation set is utilized as the test set and a subset of the train dataset is being sampled for the
    val set.

    :param cfg: A configuration dictionary containing the following parameter values: 'seed' is an integer number
    that should be used as the seed for the random module in order to support reproducibility, 'model_name' is the
    string name of the detection model, 'data_path' contains the string path to the dataset root folder,
    'min_obj_pixels' specifies the visible pixel amount threshold for filtering pedestrians, 'max_occl' specifies the
    occlusion threshold for filtering pedestrians, 'max_distance' specifies the distance threshold for filtering
    pedestrians, 'batch_size' specifies the amount of image samples in a single training batch and 'num_workers'
    specifies the amount of sub-processes for loading the data.
    :return: A train, validation and test dataloader instance, containing samples from the CityPersons dataset.
    """
    # Sets the random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    # Checks whether instance segmentation labels should be loaded
    instance_segm = 'IS' in cfg['model_name']
    # Loads the CityPersons dataset instances for training, validation and testing of the detection model
    train_dataset = CityPersons_dataset(
        cfg['data_path'],
        get_transform(train=True),  # train=True
        min_obj_pixels=cfg['min_obj_pixels'],
        max_occl=cfg['max_occl'],
        max_distance=cfg['max_distance'],
        instance_segm=instance_segm
    )
    val_dataset = CityPersons_dataset(cfg['data_path'], instance_segm=instance_segm)
    test_dataset = CityPersons_dataset(cfg['data_path'], instance_segm=instance_segm)

    # Used for storing the sample indices for the train and test samples
    # Note that the official validation set will be used for testing and a subset of the train set will be used for
    # validation
    train_indices = []
    test_indices = []

    # Iterates over all dataset samples to determine whether they belong to the train or val split
    for sample_idx, img_sample_path in enumerate(train_dataset.imgs):
        split = img_sample_path.split('/')[1]
        if split == 'train':
            train_indices.append(sample_idx)
        elif split == 'val':
            test_indices.append(sample_idx)
    # Samples randomly 300 train samples to be used for the validation set
    random.shuffle(train_indices)
    val_indices = train_indices[:301]
    train_indices = train_indices[301:]

    # Splits the dataset samples into train, validation and test samples
    train_dataset = torch.utils.data.Subset(train_dataset,
                                            train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    # Loads the train, validation and test dataset into PyTorch dataloader instances
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    return train_loader, val_loader, test_loader


def load_kia_dataloaders(cfg):
    """
    Loads the train, validation and test dataloader instances for the KIA dataset, used during the training process.
    Note that only the train dataset may contain keypoint/instance-segmentation annotations. The evaluation process
    takes only the bounding box predictions into account, hence the validation and test dataloader instances only
    contain bounding box annotations.

    :param cfg: A configuration dictionary containing the following parameter values: 'seed' is an integer number that
    should be used as the seed for the random module in order to support reproducibility, 'model_name' is the string
    name of the detection model, 'data_path' contains the string path to the dataset root folder, 'min_obj_pixels'
    specifies the visible pixel amount threshold for filtering pedestrians, 'max_occl' specifies the occlusion
    threshold for filtering pedestrians, 'max_distance' specifies the distance threshold for filtering pedestrians,
    'batch_size' specifies the amount of image samples in a single training batch and 'num_workers' specifies the amount
    of sub-processes for loading the data.
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
        get_transform(train=True),
        min_obj_pixels=cfg['min_obj_pixels'],
        max_occl=cfg['max_occl'],
        max_distance=cfg['max_distance'],
        instance_segm=instance_segm,
        keypoint_detect=keypoint_det
    )
    val_dataset = KIA_dataset(cfg['data_path'], instance_segm=instance_segm)
    test_dataset = KIA_dataset(cfg['data_path'], instance_segm=instance_segm)

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
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=cfg['num_workers'],
        collate_fn=utils.collate_fn
    )
    return train_loader, val_loader, test_loader


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


class EuroCityPersons_dataset(torch.utils.data.Dataset):
    """
        Class used for loading the EuroCity Persons dataset.
        The data is used for training 2D detection models only.
        """
    def __init__(self, root, ann_file, transforms=get_transform(False),
                 remove_images_without_annotations=False):
        """
        Instantiates the class for handling the EuroCity Persons dataset.

        :param root: path including the EuroCity Persons
        :param ann_file: paths of the annotations of EuroCity Persons
        :param transforms: data agumentation, only set to True during training!
        :param remove_images_without_annotations: If remove the image without any labels. Default False
        """
        super().__init__()

        self.transforms = transforms

        if isinstance(root, (str, bytes)):
            root = os.path.expanduser(root)
        self.root = root

        print('load annotation file: ', ann_file)
        with open(ann_file, "r") as f:
            dataset = json.load(f)

        # a dict to store the images: key is the image id and value is the image path
        self.imgs = dict()
        for img in dataset["images"]:  # a dict
            self.imgs[img["id"]] = img

        # a dict to store the annotations: key is the image id and value is a list containing annotations
        self.imgs_with_anns = defaultdict(list)  # default type is list
        for ann in dataset["annotations"]:
            self.imgs_with_anns[ann["image_id"]].append(ann)  # type(ann): a dict annotation

        # a dict stores the category of the annotations
        self.cats = dict()
        for cat in dataset["categories"]:
            self.cats[cat["id"]] = cat

        # A list contains images ids
        self.ids = list(sorted(self.imgs.keys()))

        # update the image id list to remove the images without labels
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                anno = self.imgs_with_anns[img_id]
                if len(anno) == 0:  # current image doesn't have any annotation: 0 Gts
                    del self.imgs[img_id]
                    del self.imgs_with_anns[img_id]
                else:
                    ids.append(img_id)

            self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        # key is the image ID (1-N), value is either image dict or a list of instance dicts
        img_id = self.ids[index]  # dict
        anno = self.imgs_with_anns[img_id]  # type(anno): a list of dict

        # input image
        file_name = self.imgs[img_id]["file_name"]
        path = os.path.join(self.root, file_name)
        image = Image.open(path).convert("RGB")

        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)

        # remove unwanted instances such as too small or non-pedestrians
        anno = self._select_ann(anno, image)

        if len(anno) != 0:
            # labels: pedestrians only! All set to 1
            labels = torch.ones(len(anno), dtype=torch.int64)

            # ignore property
            iscrowd = [obj["iscrowd"] for obj in anno]
            iscrowd = np.array(iscrowd, dtype=np.int32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)  # 1 ignore otherwise not ignore
            ignore = iscrowd

            # full boxes
            fboxes = [obj["bbox"] for obj in anno]
            fboxes = np.array(fboxes, dtype=np.float32).reshape(-1, 4)
            # transfer boxes from xywh to xyxy
            fboxes[:, 2:] += fboxes[:, :2]
            fboxes = torch.as_tensor(fboxes, dtype=torch.float32)

            # PLF extractor
            bbox_height_list, bbox_width_list, bbox_aspect_ratio_list, bbox_area_list, truncated_list, \
                bbox_brightness_list, bbox_contrast_list, entropy_list, crowdedness_list \
                = self.plf_extractor(fboxes.numpy(), path, image)

            # make sure that all the lists have the same length
            lists = [bbox_height_list, bbox_aspect_ratio_list, entropy_list, crowdedness_list, bbox_area_list]

            assert all(len(lists[0]) == len(l) for l in lists[1:])

            target = {
                'labels': labels,
                'boxes': fboxes,
                'area': torch.as_tensor(bbox_area_list, dtype=torch.float32),
                'ignore': ignore,
                'iscrowd': iscrowd,
                # plf values all numerical
                'bbox_height': torch.as_tensor(np.array(bbox_height_list, dtype=int), dtype=torch.int64),
                'bbox_aspect_ratio': torch.as_tensor(bbox_aspect_ratio_list, dtype=torch.float32),
                'entropy': torch.as_tensor(entropy_list, dtype=torch.float32),
                'crowdedness': torch.as_tensor(crowdedness_list, dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
            }

        else:
            target = {
                'labels': torch.zeros(0, dtype=torch.int64),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'area': torch.zeros(0, dtype=torch.float32),
                'ignore': torch.zeros(0, dtype=torch.uint8),
                'iscrowd': torch.zeros(0, dtype=torch.uint8),
                # plf values all numerical
                'bbox_height': torch.zeros(0, dtype=torch.int64),
                'bbox_aspect_ratio': torch.zeros((0, 4), dtype=torch.float32),
                'entropy': torch.zeros((0, 4), dtype=torch.float32),
                'crowdedness': torch.zeros((0, 4), dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
            }

        # Applies the preprocessing on the image sample and annotation dictionary, using the transforms function
        img, target = self.transforms(image, target)
        return img, target

    def _select_ann(self, ann_info, img):
        """
        remove the unwanted instances such as zero area instances.
        :param ann_info: the annotation dict for the current image
        :param img: the current image
        :return: the anno dict after removing the unwanted instances' annotations
        """
        gt_ann = []

        for i, ann in enumerate(ann_info):
            # discard the ignore instance
            if ann.get('ignore', False):  # False is returned when 'ignore' is not a key of ann
                continue

            # discard small instances
            x1, y1, w, h = ann['bbox']
            if w < 1 or h < 1:
                continue

            # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
            x1 = min(img.width - 1, max(0, ann['bbox'][0]))
            y1 = min(img.height - 1, max(0, ann['bbox'][1]))
            x2 = min(img.width - 1, max(0, ann['bbox'][0] + int(ann['bbox'][2])))
            y2 = min(img.height - 1, max(0, ann['bbox'][1] + int(ann['bbox'][3])))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                continue

            # discard iscrowd
            if ann['iscrowd']:
                continue

            gt_ann.append(ann)

        return gt_ann

    def plf_extractor(self, bboxes, img_path, img):
        """
        extract the PLF factors.
        For ECP dataset, available PLFs at the instance level include box height, box aspect ratio, entropy, crowdedness
        :param bboxes: the ground truth bounding boxes annotations
        :param img_path: the path of the input image
        :param img: the current input image
        :return: multiple lists storing the corresponding PLF values
        """
        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)

        # placeholder
        bbox_height_list = []  # plf, object, numerical
        bbox_width_list = []
        bbox_aspect_ratio_list = []  # plf, object, numerical
        bbox_area_list = []
        truncated_list = []
        bbox_brightness_list = []
        bbox_contrast_list = []
        entropy_list = []  # plf, object, numerical
        crowdedness_list = []  # plf, object, numerical

        for box in bboxes:  # instance level
            # Stores the bounding box information
            x1 = min(img.width - 1, max(0, box[0]))
            y1 = min(img.height - 1, max(0, box[1]))
            x2 = min(img.width - 1, max(0, box[2]))
            y2 = min(img.height - 1, max(0, box[3]))
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Checks whether the bbox coordinates lie on the edge of the image -> pedestrian is truncated
            if x1 == 0 or x1 == sample_rgb.shape[1] - 1 or \
                    y1 == 0 or y1 == sample_rgb.shape[0] - 1 or \
                    x2 == 0 or x2 == sample_rgb.shape[1] - 1 or \
                    y2 == 0 or y2 == sample_rgb.shape[0] - 1:
                truncated = True
            else:
                truncated = False

            # Extracts some bbox statistics
            bbox_height = int(y2 - y1)
            bbox_width = int(x2 - x1)
            bbox_aspect_ratio = round(bbox_width / bbox_height, 3)
            bbox_area = int(bbox_width * bbox_height)

            # Extracts some image statistics w.r. to the bbox
            bbox_gray = sample_gray[y1:y2, x1:x2]
            bbox_brightness = round(float(bbox_gray.mean()), 3) / 255 * 100
            bbox_contrast = round(float(bbox_gray.std()), 3) / 127.5 * 100

            # Measures the Shannon entropy within the bbox area
            entropy = shannon_entropy(bbox_gray)

            # append
            bbox_height_list.append(bbox_height)
            bbox_width_list.append(bbox_width)
            bbox_aspect_ratio_list.append(bbox_aspect_ratio)
            bbox_area_list.append(bbox_area)
            truncated_list.append(truncated)
            bbox_brightness_list.append(bbox_brightness)
            bbox_contrast_list.append(bbox_contrast)
            entropy_list.append(entropy)
            crowdedness_list.append(0)

        # crowdedness
        # Iterates over all possible pairs of bboxes and computes the overlap for each of them
        for box_pair in list(itertools.combinations(range(len(bboxes)), 2)):
            # Calculates the corner coordinates for the intersection of the two bboxes
            x1 = max(bboxes[box_pair[0]][0], bboxes[box_pair[1]][0])
            y1 = max(bboxes[box_pair[0]][1], bboxes[box_pair[1]][1])
            x2 = min(bboxes[box_pair[0]][2], bboxes[box_pair[1]][2])
            y2 = min(bboxes[box_pair[0]][3], bboxes[box_pair[1]][3])
            # Calculates the area for the intersection
            inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter_area == 0:
                overlap1 = 0
                overlap2 = 0
            else:
                # Computes the overlap
                overlap1 = inter_area / bbox_area_list[box_pair[0]]
                overlap2 = inter_area / bbox_area_list[box_pair[1]]

            # Uses the overlap values and scales them by the ratio between the two bboxes to compute the crowdedness
            bboxes_ratio = min(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]]) /\
                           max(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]])

            crowdedness_list[box_pair[0]] += overlap1 * bboxes_ratio
            crowdedness_list[box_pair[1]] += overlap2 * bboxes_ratio

        return bbox_height_list, bbox_width_list, bbox_aspect_ratio_list, \
            bbox_area_list, truncated_list, bbox_brightness_list, \
            bbox_contrast_list, entropy_list, crowdedness_list


class CityPersons_dataset(torch.utils.data.Dataset):
    """
    Class used for loading the CityPersons dataset. The data is used for training 2D detection models and supports also
    detection with instance segmentation.
    """

    def __init__(self, root_path, transforms=get_transform(False), min_obj_pixels=None, max_occl=None,
                 max_distance=None,
                 instance_segm=False):
        """
        Instantiates the class for handling the CityPersons dataset. Note: If the dataset instance is being used for
        evaluation (val or test split), then only the root_path needs to be set.

        :param root_path: The string path to the root folder of the CP dataset. Must be of type string.
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
        :param max_distance: Filter out all pedestrian instances (during training) that have a higher distance than here
        specified. Must be a number greater or equal to zero, that specifies the distance in meters. Set this to None
        if the data is being loaded for evaluation.
        :param instance_segm: Whether to load the instance segmentation labels. Must be boolean value. Set this only
        for training.
        """
        self.root_path = root_path
        self.transforms = transforms
        self.min_obj_pixels = min_obj_pixels
        self.max_occl = max_occl
        self.max_distance = max_distance
        self.instance_segm = instance_segm

        self.classes = {'human': 1}  # Maps the class names with their id (id 0 is reserved for the background class)

        # CityPersons dataset path postfixes
        self.img_sample_postfix = 'leftImg8bit'
        self.bbox_2D_postfix = 'gtBboxCityPersons'
        self.instance_segm_postfix = 'gtFine'
        self.depth_postfix = 'disparity'

        # Stores the relative paths (within the CityPersons dataset), to all (required) dataset files
        self.imgs, self.bbox_jsons, self.instance_segm_imgs = self.get_file_paths()

    def __len__(self):
        """
        :return: Amount of image samples.
        """
        return len(self.imgs)

    def __getitem__(self, idx):
        """
        Queries the CityPersons data at runtime, by indexing the precomputed sample list holding the relative paths to
        all image samples. The image and the corresponding labels are loaded and preprocessed by a transforms function,
        before being returned.

        :param idx: An integer value, used for indexing the CityPersons dataset. Must be an integer number between 0 and
        (sample amount - 1).
        :return: Two elements are being returned. The first one is the image sample, which is loaded by PIL and
        preprocessed by a transforms function. The second element is a target dictionary, holding the following key
        value pairs: boxes (key) -> A float tensor of size [N, 4] holding the bbox coordinates for each instance,
        labels (key) -> An integer tensor of size [N] holding the class id's for each instance, area (key) -> A float
        tensor of size [N] holding the bbox area for each instance, ignore (key) -> An integer tensor of size [N]
        holding the information whether the pedestrian instance is safety relevant (0) or should be ignored (1),
        iscrowd (key) -> An integer tensor of size [N] holding the information whether the pedestrian annotations
        represent a group of people (1) or just a single pedestrian instance (0), masks (key) -> An integer tensor of
        size [N, H, W] holding the binary instance segmentation masks for each instance.
        """
        # Path to the image sample and 2D bbox annotation json file
        img_path = f'{self.root_path}/{self.imgs[idx]}'
        img = Image.open(img_path).convert('RGB')  # Loads the image sample as a PIL Image
        # bbox
        bbox_json_path = f'{self.root_path}/{self.bbox_jsons[idx]}'
        # seg
        instance_segm_path = f'{self.root_path}/{self.instance_segm_imgs[idx]}'
        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)

        # Extracts the 2D bbox information from the annotation json file
        instances, bboxes, class_ids, bbox_areas, ignore, iscrowd, bbox_height_list, bbox_aspect_ratio_list, visible_instance_pixels_list, occlusion_ratio_list, distance_list, foreground_brightness_list, contrast_to_background_list, entropy_list, background_edge_strength_list, boundary_edge_strength_list, crowdedness_list = \
            self.extract_2D_bbox_data(bbox_json_path,
                                      img,
                                      img_path,
                                      instance_segm_path)

        # make sure that all the lists have the same length
        lists = [instances, bboxes, class_ids, bbox_areas, ignore, iscrowd, bbox_height_list, bbox_aspect_ratio_list,
                 visible_instance_pixels_list, occlusion_ratio_list, distance_list, foreground_brightness_list,
                 contrast_to_background_list, entropy_list, background_edge_strength_list, boundary_edge_strength_list,
                 crowdedness_list]

        assert all(len(lists[0]) == len(l) for l in lists[1:])

        if len(instances) != 0:
            # Stores all the 2D bbox information inside a target dictionary
            target = {
                'instance id': instances,
                'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
                'labels': torch.as_tensor(class_ids, dtype=torch.int64),
                'area': torch.as_tensor(bbox_areas, dtype=torch.float32),
                'ignore': torch.as_tensor(ignore, dtype=torch.uint8),
                'iscrowd': torch.as_tensor(iscrowd, dtype=torch.uint8),
                # plf values all numerical
                'bbox_height': torch.as_tensor(np.array(bbox_height_list, dtype=int), dtype=torch.int64),
                'bbox_aspect_ratio': torch.as_tensor(bbox_aspect_ratio_list, dtype=torch.float32),
                'visible_instance_pixels': torch.as_tensor(np.array(visible_instance_pixels_list, dtype=int),
                                                           dtype=torch.int64),
                'occlusion_ratio': torch.as_tensor(occlusion_ratio_list, dtype=torch.float32),
                'distance': torch.as_tensor(distance_list, dtype=torch.float32),
                'foreground_brightness': torch.as_tensor(foreground_brightness_list, dtype=torch.float32),
                'contrast_to_background': torch.as_tensor(np.array(contrast_to_background_list, dtype=float),
                                                          dtype=torch.float32),
                'entropy': torch.as_tensor(entropy_list, dtype=torch.float32),
                'background_edge_strength': torch.as_tensor(np.array(background_edge_strength_list, dtype=float),
                                                            dtype=torch.float32),
                'boundary_edge_strength': torch.as_tensor(np.array(boundary_edge_strength_list, dtype=float),
                                                          dtype=torch.float32),
                'crowdedness': torch.as_tensor(crowdedness_list, dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
            }

            if self.instance_segm:
                # Path to the instance segmentation image
                instance_img_path = f'{self.root_path}/{self.instance_segm_imgs[idx]}'
                instance_segm_masks = self.extract_instance_segm_masks(instance_img_path, instances)
                target['masks'] = torch.as_tensor(instance_segm_masks, dtype=torch.uint8)
        else:
            # Stores the target information for an image samsple without any objects
            target = {
                'instance id': [],
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "ignore": torch.zeros(0, dtype=torch.uint8),
                "iscrowd": torch.zeros(0, dtype=torch.uint8),
                # plf values all numerical
                'bbox_height': torch.zeros(0, dtype=torch.int64),
                'bbox_aspect_ratio': torch.zeros((0, 4), dtype=torch.float32),
                'visible_instance_pixels': torch.zeros(0, dtype=torch.int64),
                'occlusion_ratio': torch.zeros((0, 4), dtype=torch.float32),
                'distance': torch.zeros((0, 4), dtype=torch.float32),
                'foreground_brightness': torch.zeros((0, 4), dtype=torch.float32),
                'contrast_to_background': torch.zeros((0, 4), dtype=torch.float32),
                'entropy': torch.zeros((0, 4), dtype=torch.float32),
                'background_edge_strength': torch.zeros((0, 4), dtype=torch.float32),
                'boundary_edge_strength': torch.zeros((0, 4), dtype=torch.float32),
                'crowdedness': torch.zeros((0, 4), dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
            }
            if self.instance_segm:
                target["masks"] = torch.zeros((0, img.height, img.width), dtype=torch.uint8)

        # Applies the preprocessing on the image sample and annotation dictionary, using the transforms function
        img, target = self.transforms(img, target)
        return img, target

    def get_file_paths(self):
        """
        Extracts the relative paths (within the CityPersons dataset) to the dataset files. Every (relative) path is made
        up from the following components: {postfix (depending on the type of data)}/{split name}/{city name}/{file name}.
        Note that only the train and validation split are extracted since the test split does not contain public
        annotations, and hence will not be used here.

        :return: Three lists containing the relative paths (within the CityPersons root folder) to every: image sample,
        bbox annotation json file and instance segmentation image. Every sample must have a consistent index within all
        lists, so it can be queried accordingly.
        """
        # Every sample must have a consistent index within all lists
        img_file_paths = []  # Relative paths to the image samples
        bbox_json_file_paths = []  # Relative paths to the bounding box annotation json files
        instance_img_file_paths = []  # Relative paths to the instance segmentation images

        for split in sorted(['train', 'val']):  # Iterates over all splits
            city_folders = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/{split}'))
            for city in city_folders:  # Iterates over all city folders
                file_names = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/{split}/{city}'))
                for file in file_names:  # Iterates over all files
                    raw_file_name = '_'.join(file.split('_')[:-1])
                    # Defines the relative paths to the dataset files
                    img_sample_path = \
                        f'{self.img_sample_postfix}/{split}/{city}/{raw_file_name}_leftImg8bit.png'
                    bbox_2D_path = \
                        f'{self.bbox_2D_postfix}/{split}/{city}/{raw_file_name}_gtBboxCityPersons.json'
                    instance_segm_path = \
                        f'{self.instance_segm_postfix}/{split}/{city}/{raw_file_name}_gtFine_instanceIds.png'

                    # Adds the file path information
                    img_file_paths.append(img_sample_path)
                    bbox_json_file_paths.append(bbox_2D_path)
                    instance_img_file_paths.append(instance_segm_path)
        return img_file_paths, bbox_json_file_paths, instance_img_file_paths

    def extract_2D_bbox_data(self, json_path, img, img_path, instance_segm_path):
        """
        Loads the 2D bbox annotation json file and extracts the pedestrian information from it.

        :param json_path: Path to the annotation json file.
        :param img: A PIL Image instance of the sample image, corresponding to the json annotation file.
        :return: A list of instance id's (string), a list of lists containing bounding box coordinates [x1, y1, x2, y2]
        as integer values, a list of class id's (integer), a list of bounding box areas (integer), a binary list of
        zeros and ones representing whether the pedestrian instance can be ignored during evaluation (not safety
        relevant) and a binary list of zeros and ones representing whether the pedestrian annotation represents a group
        of pedestrians (iscrowd=1) or a single pedestrian instance (iscrowd=0).
        """
        # bbox annotation
        with open(json_path) as json_file:
            labels = json.load(json_file)  # Loads the annotations from the json file
        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
        # Loads the instance segmentation image
        instance_segm = cv2.imread(instance_segm_path, cv2.IMREAD_UNCHANGED)

        # Placeholder for storing the annotation information
        bboxes = []
        instance_id_list = []
        label_list = []
        bounding_box_list = []
        mask_list = []
        citypersons_label_list = []
        bbox_height_list = []  # plf, object, numerical
        bbox_width_list = []
        bbox_aspect_ratio_list = []  # plf, object, numerical
        bbox_area_list = []
        visible_instance_pixels_list = []  # plf, object, numerical
        foreground_to_bbox_ratio_list = []
        occlusion_ratio_list = []  # plf, object, numerical
        truncated_list = []
        distance_list = []  # plf, object, numerical
        bbox_brightness_list = []
        bbox_contrast_list = []
        foreground_brightness_list = []  # plf, object, numerical
        foreground_contrast_list = []
        background_brightness_list = []
        background_contrast_list = []
        contrast_to_background_list = []  # plf, object, numerical
        entropy_list = []  # plf, object, numerical
        foreground_edge_strength_list = []
        background_edge_strength_list = []  # plf, object, numerical
        boundary_edge_strength_list = []  # plf, object, numerical
        crowdedness_list = []  # plf, object, numerical
        ignore_list = []
        iscrowd_list = []

        # Iterates over all annotated instances from the bbox annotation json file
        for obj in labels['objects']:
            # Filters out objects that are not of the desired pedestrian class
            if obj['label'] != 'ignore':
                # Stores the bounding box information (converted to the corner format and clamped)
                x1 = min(img.width - 1, max(0, obj['bbox'][0]))
                y1 = min(img.height - 1, max(0, obj['bbox'][1]))
                x2 = min(img.width - 1, max(0, obj['bbox'][0] + int(obj['bbox'][2])))
                y2 = min(img.height - 1, max(0, obj['bbox'][1] + int(obj['bbox'][3])))
                if (x2 - x1) * (y2 - y1) == 0 or x1 >= x2 or y1 >= y2:
                    # Skips the instance if the bbox area is zero or the bbox coordinates are inconsistent
                    continue
                if obj['instance_pixels'] < 1 or obj['occlusion_est'] > 0.99:
                    # Skips the instance if it is not visible at all
                    continue
                # Checks whether the data is being loaded for training or for evaluation (everything set to None)
                if self.min_obj_pixels is not None and self.max_occl is not None and self.max_distance is not None:
                    # Filters out objects that are too occluded, too small or too far away (only for training)
                    if (obj['instance_pixels'] < self.min_obj_pixels or
                            obj['occlusion_est'] > self.max_occl or
                            obj['distance'] > self.max_distance):
                        continue

                bboxes.append([x1, y1, x2, y2])  # Stores the original bbox coordinates

                # Checks whether the bbox coordinates lie on the edge of the image -> pedestrian is truncated
                if x1 == 0 or x1 == sample_rgb.shape[1] - 1 or \
                        y1 == 0 or y1 == sample_rgb.shape[0] - 1 or \
                        x2 == 0 or x2 == sample_rgb.shape[1] - 1 or \
                        y2 == 0 or y2 == sample_rgb.shape[0] - 1:
                    truncated = True
                else:
                    truncated = False

                instance_id = str(obj['instanceId'])

                citypersons_label = obj['label']
                # Extracts some bbox statistics
                bbox_height = int(y2 - y1)
                bbox_width = int(x2 - x1)
                bbox_aspect_ratio = round(bbox_width / bbox_height, 3)
                bbox_area = int(bbox_width * bbox_height)
                visible_instance_pixels = int(obj['instance_pixels'])
                foreground_to_bbox_ratio = round(visible_instance_pixels / bbox_area, 3)
                occlusion_ratio = float(obj['occlusion_est'])
                distance = float(obj['distance'])
                ignore = distance > 50  # Ignore instances with a distance > 50m
                iscrowd = bool(obj['label'] == 'person group')

                # Extracts the instance segmentation mask
                mask = cv2.inRange(instance_segm[y1:y2, x1:x2], obj['instanceId'], obj['instanceId']) / 255
                # Skips the instance if the instance segmentation mask does not exist
                if not np.any(mask.astype(np.uint8)):
                    # Removes the previously added bbox coordinates
                    bboxes.pop()
                    continue

                # Extracts some image statistics w.r. to the bbox
                bbox_gray = sample_gray[y1:y2, x1:x2]
                bbox_brightness = round(float(bbox_gray.mean()), 3) / 255 * 100
                bbox_contrast = round(float(bbox_gray.std()), 3) / 127.5 * 100
                foreground_brightness = round(float(bbox_gray[np.nonzero(mask)].mean()), 3) / 255 * 100
                foreground_contrast = round(float(bbox_gray[np.nonzero(mask)].std()), 3) / 127.5 * 100
                background_mask = 1 - mask
                if np.any(background_mask):
                    background_brightness = round(float(bbox_gray[np.nonzero(background_mask)].mean()),
                                                  3) / 255 * 100
                    background_contrast = round(float(bbox_gray[np.nonzero(background_mask)].std()),
                                                3) / 127.5 * 100
                    contrast_to_background = round(abs(background_contrast - foreground_contrast), 3)
                else:
                    background_brightness = None
                    background_contrast = None
                    contrast_to_background = None

                # Measures the Shannon entropy within the bbox area
                entropy = shannon_entropy(bbox_gray)

                # Quantifies the edge strength
                # Applies the dilation and erosion operation to create two new masks
                kernel = np.ones((2, 2), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # larger than original mask
                eroded_mask = cv2.erode(mask, kernel, iterations=2)  # smaller than original mask
                boundary_mask = dilated_mask - eroded_mask
                # Calculates the edge magnitude for the bbox
                magnitude = utils.get_edge_magnitude(bbox_gray)
                # Masks out the magnitude using the dilated mask
                foreground_magnitude = magnitude[np.nonzero(dilated_mask)]
                # Masks out the magnitude using the inverted dilated mask
                background_magnitude = magnitude[np.nonzero(1 - dilated_mask)]
                # Masks out the magnitude using the boundary mask
                boundary_magnitude = magnitude[np.nonzero(boundary_mask)]
                # Calculates the edge strength for the foreground, background and boundary by taking the mean
                foreground_edge_strength = round(float(foreground_magnitude.mean()), 3)
                if np.any(background_magnitude):
                    background_edge_strength = round(float(background_magnitude.mean()), 3)
                else:
                    background_edge_strength = None
                if np.any(boundary_magnitude):
                    boundary_edge_strength = round(float(boundary_magnitude.mean()), 3)
                else:
                    boundary_edge_strength = None

                instance_id_list.append(instance_id)
                label_list.append(1)
                mask_list.append(mask)
                citypersons_label_list.append(citypersons_label)
                bbox_height_list.append(bbox_height)
                bbox_width_list.append(bbox_width)
                bbox_aspect_ratio_list.append(bbox_aspect_ratio)
                bbox_area_list.append(bbox_area)
                visible_instance_pixels_list.append(visible_instance_pixels)
                foreground_to_bbox_ratio_list.append(foreground_to_bbox_ratio)
                occlusion_ratio_list.append(occlusion_ratio)
                truncated_list.append(truncated)
                distance_list.append(distance)
                bbox_brightness_list.append(bbox_brightness)
                bbox_contrast_list.append(bbox_contrast)
                foreground_brightness_list.append(foreground_brightness)
                foreground_contrast_list.append(foreground_contrast)
                background_brightness_list.append(background_brightness)
                background_contrast_list.append(background_contrast)
                contrast_to_background_list.append(contrast_to_background)
                entropy_list.append(entropy)
                foreground_edge_strength_list.append(foreground_edge_strength)
                background_edge_strength_list.append(background_edge_strength)
                boundary_edge_strength_list.append(boundary_edge_strength)
                crowdedness_list.append(0)
                ignore_list.append(ignore)
                iscrowd_list.append(iscrowd)

        # Iterates over all possible pairs of bboxes and computes the overlap for each of them
        for box_pair in list(itertools.combinations(range(len(bboxes)), 2)):
            # Calculates the corner coordinates for the intersection of the two bboxes
            x1 = max(bboxes[box_pair[0]][0], bboxes[box_pair[1]][0])
            y1 = max(bboxes[box_pair[0]][1], bboxes[box_pair[1]][1])
            x2 = min(bboxes[box_pair[0]][2], bboxes[box_pair[1]][2])
            y2 = min(bboxes[box_pair[0]][3], bboxes[box_pair[1]][3])
            # Calculates the area for the intersection
            inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter_area == 0:
                overlap1 = 0
                overlap2 = 0
            else:
                # Computes the overlap
                overlap1 = inter_area / bbox_area_list[box_pair[0]]
                overlap2 = inter_area / bbox_area_list[box_pair[1]]
                # Updates the ignore list, if one of the two bboxes is being overlapped by 60% or more (heavily crowded)
                if overlap1 >= 0.6 or overlap2 >= 0.6:
                    # Determines which of the two bboxes has a higher distance
                    if distance_list[box_pair[0]] > distance_list[box_pair[1]]:
                        ignore_list[box_pair[0]] = 1  # instance with higher distance is ignored
                    else:
                        ignore_list[box_pair[1]] = 1

            # Uses the overlap values and scales them by the ratio between the two bboxes to compute the crowdedness
            bboxes_ratio = min(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]]) / \
                           max(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]])
            crowdedness_list[box_pair[0]] += overlap1 * bboxes_ratio
            crowdedness_list[box_pair[1]] += overlap2 * bboxes_ratio

        return instance_id_list, bboxes, label_list, bbox_area_list, ignore_list, iscrowd_list, bbox_height_list, bbox_aspect_ratio_list, visible_instance_pixels_list, occlusion_ratio_list, distance_list, foreground_brightness_list, contrast_to_background_list, entropy_list, background_edge_strength_list, boundary_edge_strength_list, crowdedness_list

    def extract_instance_segm_masks(self, img_path, instances):
        """
        Loads the instance segmentation image and extracts the binary instance segmentation masks.

        :param img_path: Path to the instance segmentation image.
        :param instances: A list of instance id's (strings), generated by the extract_2D_bbox_data function.
        :return: A numpy list of binary instance segmentation masks in numpy format.
        """
        # Loads the instance segmentation image
        instance_segm_img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        instance_masks = []  # Used for storing the binary instance segmentation masks for each bbox instance
        # Iterates over all bbox instances to produce the corresponding instance segmentation mask
        for instance_id in instances:
            # Produces a binary mask for the given instance id
            mask = cv2.inRange(instance_segm_img, int(instance_id), int(instance_id)) / 255
            instance_masks.append(mask)  # Stores the produced mask for the current instance
        return np.array(instance_masks)

    def preprocess_raw_CityPersons(self, occl_model_path):
        """
        This function should be called right after the raw CityPersons data has been downloaded to process it for the
        first time. The function loads the depth maps for each sample image to extract the distance information for
        each pedestrian instance. The information about the amount of visible instance pixels is extracted from the
        instance segmentation mask by counting. For estimating the occlusion ratio, a regression model is being used
        which was previously trained on the fully annotated samples from the synthetic KIA dataset, to predict the
        occlusion ratio based on the bounding box area, the amount of visible instance pixels, the ratio of empty
        bounding box rows and the ratio of empty bounding box columns (compared to the total amount of bounding box rows
        and columns). The information about the distance, the amount of visible instance pixels and the occlusion ratio
        is finally stored within the original 2D bbox json annotation files (gtBboxCityPersons).

        :param occl_model_path: String path to a pickle file, which contains the occlusion estimation regression model.
        """
        # Iterates over all splits
        for split in tqdm.tqdm(['train', 'val'], desc='Loading Split', position=0):
            city_folders = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/{split}'))
            # Iterates over all city folders
            for city in tqdm.tqdm(city_folders, desc='Loading Cities', position=1, leave=False, postfix=split):
                file_names = sorted(os.listdir(f'{self.root_path}/{self.bbox_2D_postfix}/{split}/{city}'))
                # Iterates over all files
                for file in tqdm.tqdm(file_names, desc='Loading Files', position=2, leave=False, postfix=city):
                    raw_file_name = '_'.join(file.split('_')[:-1])  # Removes the postfix to extract the raw file name

                    # Defines the paths to the dataset files
                    bbox_2D_json_path = \
                        f'{self.root_path}/{self.bbox_2D_postfix}/{split}/{city}/{raw_file_name}_gtBboxCityPersons.json'
                    instance_segm_path = \
                        f'{self.root_path}/{self.instance_segm_postfix}/{split}/{city}/{raw_file_name}_gtFine_instanceIds.png'
                    depth_map_path = \
                        f'{self.root_path}/{self.depth_postfix}/{split}/{city}/{raw_file_name}_disparity.png'

                    # Loads and processes the depth map
                    depth_img = cv2.imread(depth_map_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    depth_img[depth_img > 0] = (depth_img[depth_img > 0] - 1) / 256
                    # Loads the instance segmentation image
                    instance_segm_img = cv2.imread(instance_segm_path, cv2.IMREAD_UNCHANGED)

                    # Loads the regression model for estimating the occlusion ratio
                    with open(occl_model_path, 'rb') as pickle_file:
                        model_file = pickle.load(pickle_file)
                        model = model_file['model']

                    with open(bbox_2D_json_path, 'r+') as json_file:
                        bbox_labels = json.load(json_file)  # Loads the annotations from the json file
                        for i, obj in enumerate(bbox_labels['objects']):
                            if obj['label'] == 'ignore':
                                # Skip non-pedestrian instances
                                continue

                            # Extracts the bounding box coordinates in corner format
                            x1 = min(2048 - 1, max(0, obj['bbox'][0]))
                            y1 = min(1024 - 1, max(0, obj['bbox'][1]))
                            x2 = min(2048 - 1, max(0, obj['bbox'][0] + int(obj['bbox'][2])))
                            y2 = min(1024 - 1, max(0, obj['bbox'][1] + int(obj['bbox'][3])))
                            # Loads the instance segmentation mask
                            mask = cv2.inRange(instance_segm_img, obj['instanceId'], obj['instanceId']) / 255
                            # Counts the amount of visible instance pixels
                            visible_instance_pixels = np.count_nonzero(mask[y1:y2, x1:x2])
                            # Computes the distance to the pedestrian
                            masked_depth = depth_img.copy() * mask.copy()
                            distance = round((0.209313 * 2262.52) / np.mean(masked_depth[np.nonzero(masked_depth)]), 2)
                            # Computes the bounding box parameters for estimating the occlusion ratio
                            bbox_area = int((x2 - x1) * (y2 - y1))
                            bbox_instance_mask = mask[y1:y2, x1:x2]
                            empty_rows = sum(~np.any(bbox_instance_mask, axis=1))
                            empty_columns = sum(~np.any(bbox_instance_mask, axis=0))
                            empty_rows_ratio = round(empty_rows / (y2 - y1), 2)
                            empty_columns_ratio = round(empty_columns / (x2 - x1), 2)
                            estimation_params = np.array(
                                [
                                    bbox_area,
                                    visible_instance_pixels,
                                    empty_rows_ratio,
                                    empty_columns_ratio
                                ]
                            )
                            # Approximate the occlusion_est
                            occlusion_est = round(np.clip(model.predict([estimation_params]).item(), 0.0, 0.99), 2)
                            bbox_labels['objects'][i]['instance_pixels'] = visible_instance_pixels
                            bbox_labels['objects'][i]['occlusion_est'] = occlusion_est
                            bbox_labels['objects'][i]['distance'] = distance

                        # Overwrites the 2D bbox annotations inside the json file
                        json_file.seek(0)
                        json_file.write(json.dumps(bbox_labels, indent=4))
                        json_file.truncate()


class KIA_dataset(torch.utils.data.Dataset):
    """
    Class used for loading the synthetic dataset from the KIA project. The data is used for training 2D detection models
    and supports also detection with instance segmentation or keypoint detection.
    """

    def __init__(self, root_path, transforms=get_transform(False), min_obj_pixels=None, max_occl=None,
                 max_distance=None,
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
        self.imgs, self.bbox_jsons, self.instance_segm_imgs, self.keypoint_jsons, self.file_names = self.get_file_paths()

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
        masks (key) -> An integer tensor of size [N, H, W] holding the binary instance segmentation masks for each
        instance, keypoints (key) -> A float tensor of size [N, K, 3] (K is the amount of annotated keypoints) holding
        the keypoint coordinates and visibility information for each keypoint.
        """
        # Path to the image sample and 2D bbox annotation json file
        img_path = f'{self.root_path}/{self.imgs[idx]}'
        bbox_json_path = f'{self.root_path}/{self.bbox_jsons[idx]}'
        instance_segm_path = f'{self.root_path}/{self.instance_segm_imgs[idx]}'
        img = Image.open(img_path).convert('RGB')  # Loads the image sample as a PIL Image

        # folder info
        sample_info = self.imgs[idx].split('/')
        company_name = sample_info[0].split('_')[2]
        tranche = sample_info[0].split('_')[-1]

        sequence = int(sample_info[1].split('_')[3].split('-')[0].lstrip(
            '0'))  # A set of characters to remove as leading characters

        file_name = self.file_names[idx]
        raw_file_name = '-'.join(file_name.split('-')[2:])  # Removes the 'arb-camera001-' prefix
        data_path = '/'.join(img_path.split('/')[:7])
        general_frame_json_path = f'{data_path}/{self.general_frame_postfix}/world-{raw_file_name}.json'
        # print(general_frame_json_path)

        # Loads the general-globally-per-frame-analysis_json annotations, which contain meta-data for tranches 6 & 7
        pedestrian_meta_data = {}
        if tranche == '6' or tranche == '7':
            with open(general_frame_json_path) as frame_json_file:
                frame_labels = json.load(frame_json_file)  # Loads the annotations from the json file
            if tranche == '7':
                # Loads pedestrian meta-data for tranche 7
                for entity in frame_labels['base_context']['entities']:
                    if 'class_id' in entity:
                        if entity['class_id'] == 'human':
                            pedestrian_meta_data[str(entity['instance_id'])] = {
                                'ood': entity['ood_asset'],
                                'mocap': entity['mocap_asset']}

        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)

        # Extracts the 2D bbox information from the annotation json file -> this is a single json file for this
        # pedestrian
        instances, bboxes, class_ids, bbox_areas, ignore, bbox_height_list, bbox_aspect_ratio_list, visible_instance_pixels_list, occlusion_ratio_list, distance_list, foreground_brightness_list, contrast_to_background_list, entropy_list, background_edge_strength_list, boundary_edge_strength_list, crowdedness_list = \
            self.extract_2D_bbox_data(bbox_json_path,
                                      img,
                                      img_path,
                                      instance_segm_path,
                                      )

        # assert the consistance of length of lists
        lists = [instances, bboxes, class_ids, bbox_areas, ignore, bbox_height_list, bbox_aspect_ratio_list,
                 visible_instance_pixels_list, occlusion_ratio_list, distance_list, foreground_brightness_list,
                 contrast_to_background_list, entropy_list, background_edge_strength_list, boundary_edge_strength_list,
                 crowdedness_list]
        assert all(len(lists[0]) == len(l) for l in lists[1:])

        if len(instances) != 0:
            # Stores all the 2D bbox information inside a target dictionary
            target = {
                'instance id': instances,
                'boxes': torch.as_tensor(bboxes, dtype=torch.float32),
                'labels': torch.as_tensor(class_ids, dtype=torch.int64),
                'area': torch.as_tensor(bbox_areas, dtype=torch.float32),
                'ignore': torch.as_tensor(ignore, dtype=torch.uint8),
                'iscrowd': torch.as_tensor([0] * len(instances), dtype=torch.uint8),
                # plf values all numerical
                'bbox_height': torch.as_tensor(np.array(bbox_height_list, dtype=int), dtype=torch.int64),
                'bbox_aspect_ratio': torch.as_tensor(bbox_aspect_ratio_list, dtype=torch.float32),
                'visible_instance_pixels': torch.as_tensor(np.array(visible_instance_pixels_list, dtype=int),
                                                           dtype=torch.int64),
                'occlusion_ratio': torch.as_tensor(occlusion_ratio_list, dtype=torch.float32),
                'distance': torch.as_tensor(distance_list, dtype=torch.float32),
                'foreground_brightness': torch.as_tensor(foreground_brightness_list, dtype=torch.float32),
                'contrast_to_background': torch.as_tensor(np.array(contrast_to_background_list, dtype=float),
                                                          dtype=torch.float32),
                'entropy': torch.as_tensor(entropy_list, dtype=torch.float32),
                'background_edge_strength': torch.as_tensor(np.array(background_edge_strength_list, dtype=float),
                                                            dtype=torch.float32),
                'boundary_edge_strength': torch.as_tensor(np.array(boundary_edge_strength_list, dtype=float),
                                                          dtype=torch.float32),
                'crowdedness': torch.as_tensor(crowdedness_list, dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
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
                'instance id': [],
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.int64),
                "area": torch.zeros(0, dtype=torch.float32),
                "ignore": torch.zeros(0, dtype=torch.uint8),
                "iscrowd": torch.zeros(0, dtype=torch.uint8),
                # plf values all numerical
                'bbox_height': torch.zeros(0, dtype=torch.int64),
                'bbox_aspect_ratio': torch.zeros((0, 4), dtype=torch.float32),
                'visible_instance_pixels': torch.zeros(0, dtype=torch.int64),
                'occlusion_ratio': torch.zeros((0, 4), dtype=torch.float32),
                'distance': torch.zeros((0, 4), dtype=torch.float32),
                'foreground_brightness': torch.zeros((0, 4), dtype=torch.float32),
                'contrast_to_background': torch.zeros((0, 4), dtype=torch.float32),
                'entropy': torch.zeros((0, 4), dtype=torch.float32),
                'background_edge_strength': torch.zeros((0, 4), dtype=torch.float32),
                'boundary_edge_strength': torch.zeros((0, 4), dtype=torch.float32),
                'crowdedness': torch.zeros((0, 4), dtype=torch.float32),
                # sample level plfs
                'brightness': round(float(sample_gray.mean()) / 255 * 100, 3),
                'contrast': round(float(sample_gray.std()) / 127.5 * 100, 3),
                'edge_strength': round(float(utils.get_edge_magnitude(sample_gray).mean()) / 255 * 100, 3)
            }
            if self.instance_segm:
                target["masks"] = torch.zeros((0, img.height, img.width), dtype=torch.uint8)
            if self.keypoint_detect:
                target["keypoints"] = torch.zeros((0, self.amount_keypoints, 3), dtype=torch.float32)

        # exclusive factors of KIA dataset
        if tranche == '6' or tranche == '7':
            # daytime_type: day, medium, low
            target['daytime_type'] = frame_labels['base_context']['light_source']['elevation']
            # sky_type: clear, low partly clouded, low completely covered
            target['sky_type'] = frame_labels['base_context']['light_source']['sky']
            # fog_intensity: factor between 0 and 100
            target['fog_intensity'] = frame_labels['base_context']['additional_fields']['height_fog'][
                                          'density'] * 100
            # Extracts some annotations, which are only present for tranche 7
            if tranche == '7':
                # sun_visible: True or False
                target['sun_visible'] = \
                    bool(frame_labels['base_context']['additional_fields']['scene_light']['sun_visible'])
                # vignette_intensity: factor between 0 and 100
                target['vignette_intensity'] = \
                    frame_labels['base_context']['additional_fields']['vignette']['intensity'] * 100
                # wetness_type: dry, slightly moist, wet with puddles
                target['wetness_type'] = frame_labels['base_context']['additional_fields']['wetness'][
                    'wetness_zwicky']
                # wetness_intensity: factor between 0 and 100
                target['wetness_intensity'] = \
                    max(frame_labels['base_context']['additional_fields']['wetness']['wetness'], 0) * 100
                # puddles_intensity: factor between 0 and 100
                target['puddles_intensity'] = \
                    max(frame_labels['base_context']['additional_fields']['wetness']['puddles'], 0) * 100
                if target['sun_visible']:
                    # Lens flare is only visible if the sun is visible
                    target['lens_flare_intensity'] = \
                        frame_labels['base_context']['additional_fields']['lens_flares']['intensity'] * 100
                else:
                    target['lens_flare_intensity'] = 0
        else:
            # Annotations below are unknown, since they were only introduced in the newer tranches
            target['daytime_type'] = None  # plf
            target['sky_type'] = None  # plf
            target['fog_intensity'] = float('nan')  # plf
            target['vignette_intensity'] = float('nan')  # plf
            target['wetness_type'] = None  # plf
            target['wetness_intensity'] = float('nan')
            target['puddles_intensity'] = float('nan')
            target['lens_flare_intensity'] = float('nan')  # plf

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
        file_names_list = []

        tranches = sorted(os.listdir(f'{self.root_path}'))
        for tranche in tranches:  # Iterates over all tranches
            sequences = sorted(os.listdir(f'{self.root_path}/{tranche}'))
            for sequence in sequences:  # Iterates over all sequences
                file_names = sorted(os.listdir(f'{self.root_path}/{tranche}/{sequence}/{self.img_sample_postfix}'))
                for file in file_names:  # Iterates over all files
                    data_path = f'{tranche}/{sequence}'  # Relative path to a specific sequence within a tranche
                    file_name = file[:-4]  # Removes the .png at the end of the file name
                    # Defines the relative paths to the dataset files
                    img_sample_path = f'{data_path}/{self.img_sample_postfix}/{file_name}.png'  # f'{tranche}/{sequence}/'sensor/camera/left/png'/{file_name}.png'
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
                    file_names_list.append(file_name)

                    # Checks whether the keypoint labels exist
                    if os.path.exists(f'{self.root_path}/{keypoint_2D_path}'):
                        keypoint_json_file_paths.append(keypoint_2D_path)
                    else:
                        keypoint_json_file_paths.append(None)
        return img_file_paths, bbox_json_file_paths, instance_img_file_paths, keypoint_json_file_paths, file_names_list

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

    def extract_2D_bbox_data(self, json_path, img, img_sample_path, instance_segm_path):
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

        # Loads the sample image in RGB and grayscale
        sample_rgb = cv2.cvtColor(cv2.imread(img_sample_path), cv2.COLOR_BGR2RGB)
        sample_gray = cv2.cvtColor(sample_rgb, cv2.COLOR_RGB2GRAY)
        # Loads the instance segmentation image in RGB
        instance_segm_rgb = cv2.cvtColor(cv2.imread(instance_segm_path), cv2.COLOR_BGR2RGB)

        # Placeholder for storing the annotation information
        bboxes = []
        instance_id_list = []
        label_list = []
        mask_list = []
        bbox_height_list = []  # plf, object, numerical
        bbox_width_list = []
        bbox_aspect_ratio_list = []  # plf, object, numerical
        bbox_area_list = []
        visible_instance_pixels_list = []  # plf, object, numerical
        foreground_to_bbox_ratio_list = []
        occlusion_ratio_list = []  # plf, object, numerical
        truncated_list = []
        distance_list = []  # plf, object, numerical
        bbox_brightness_list = []
        bbox_contrast_list = []
        foreground_brightness_list = []  # plf, object, numerical
        foreground_contrast_list = []
        background_brightness_list = []
        background_contrast_list = []
        contrast_to_background_list = []  # plf, object, numerical
        entropy_list = []  # plf, object, numerical
        foreground_edge_strength_list = []
        background_edge_strength_list = []  # plf, object, numerical
        boundary_edge_strength_list = []  # plf, object, numerical
        crowdedness_list = []  # plf, object, numerical
        ignore_list = []

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
                            labels[obj]['occlusion_est'] > self.max_occl or
                            min(labels[obj]['distance'], 100) > self.max_distance):
                        continue

                # Checks whether the bbox coordinates lie on the edge of the image -> pedestrian is truncated
                if x1 == 0 or x1 == sample_rgb.shape[1] - 1 or \
                        y1 == 0 or y1 == sample_rgb.shape[0] - 1 or \
                        x2 == 0 or x2 == sample_rgb.shape[1] - 1 or \
                        y2 == 0 or y2 == sample_rgb.shape[0] - 1:
                    truncated = True
                else:
                    truncated = False

                # Extracts the instance segmentation mask
                # Converts the instance id into a hexadecimal color code
                hex_color_encoding = hex(int(obj)).replace('0x', '').zfill(6)
                # Converts the hexadecimal color code into an RGB color code
                rgb_color = np.array([int(hex_color_encoding[i:i + 2], 16) for i in (0, 2, 4)])
                # Cuts out the bbox from the instance segmentation mask
                mask = instance_segm_rgb[y1:y2, x1:x2]
                # Produces a binary mask for the given RGB color code
                mask = cv2.inRange(mask, rgb_color, rgb_color) / 255

                # Skips the instance if the instance segmentation mask does not exist
                if not np.any(mask.astype(np.uint8)):
                    continue

                # start appending!
                bboxes.append([x1, y1, x2, y2])  # Stores the bbox coordinates

                truncated_list.append(int(truncated))

                instance_id_list.append(obj)  # Stores the instance id
                label_list.append(self.classes[labels[obj]['class_id']])  # Stores the class information

                # Extracts some bbox statistics
                bbox_height = int(y2 - y1)
                bbox_width = int(x2 - x1)
                bbox_width_list.append(bbox_width)
                bbox_height_list.append(bbox_height)
                bbox_aspect_ratio_list.append(round(bbox_width / bbox_height, 3))
                bbox_area = int(bbox_width * bbox_height)
                bbox_area_list.append(bbox_area)  # Stores the bounding box area information
                visible_instance_pixels = int(labels[obj]['instance_pixels'])
                visible_instance_pixels_list.append(visible_instance_pixels)
                foreground_to_bbox_ratio_list.append(round(visible_instance_pixels / bbox_area, 3))
                occlusion_ratio_list.append(float(labels[obj]['occlusion_est']))
                distance_list.append(float(labels[obj]['distance']))
                ignore_list.append(int(labels[obj]['distance'] > 50))  # Ignore instances with a distance > 50m

                mask_list.append(mask)
                # Extracts some image statistics w.r. to the bbox
                bbox_gray = sample_gray[y1:y2, x1:x2]
                bbox_brightness = round(float(bbox_gray.mean()), 3) / 255 * 100
                bbox_contrast = round(float(bbox_gray.std()), 3) / 127.5 * 100
                foreground_brightness = round(float(bbox_gray[np.nonzero(mask)].mean()), 3) / 255 * 100
                foreground_contrast = round(float(bbox_gray[np.nonzero(mask)].std()), 3) / 127.5 * 100

                bbox_brightness_list.append(bbox_brightness)
                bbox_contrast_list.append(bbox_contrast)
                foreground_brightness_list.append(foreground_brightness)
                foreground_contrast_list.append(foreground_contrast)

                background_mask = 1 - mask
                if np.any(background_mask):
                    background_brightness = round(float(bbox_gray[np.nonzero(background_mask)].mean()), 3) / 255 * 100
                    background_contrast = round(float(bbox_gray[np.nonzero(background_mask)].std()), 3) / 127.5 * 100
                    contrast_to_background = round(abs(background_contrast - foreground_contrast), 3)
                else:
                    background_brightness = None
                    background_contrast = None
                    contrast_to_background = None

                background_brightness_list.append(background_brightness)
                background_contrast_list.append(background_contrast)
                contrast_to_background_list.append(contrast_to_background)

                # Measures the Shannon entropy within the bbox area
                entropy = shannon_entropy(bbox_gray)
                entropy_list.append(entropy)

                # Quantifies the edge strength
                # Applies the dilation and erosion operation to create two new masks
                kernel = np.ones((2, 2), np.uint8)
                dilated_mask = cv2.dilate(mask, kernel, iterations=2)  # larger than original mask
                eroded_mask = cv2.erode(mask, kernel, iterations=2)  # smaller than original mask
                boundary_mask = dilated_mask - eroded_mask
                # Calculates the edge magnitude for the bbox
                magnitude = utils.get_edge_magnitude(bbox_gray)
                # Masks out the magnitude using the dilated mask
                foreground_magnitude = magnitude[np.nonzero(dilated_mask)]
                # Masks out the magnitude using the inverted dilated mask
                background_magnitude = magnitude[np.nonzero(1 - dilated_mask)]
                # Masks out the magnitude using the boundary mask
                boundary_magnitude = magnitude[np.nonzero(boundary_mask)]
                # Calculates the edge strength for the foreground, background and boundary by taking the mean
                foreground_edge_strength = round(float(foreground_magnitude.mean()), 3)
                if np.any(background_magnitude):
                    background_edge_strength = round(float(background_magnitude.mean()), 3)
                else:
                    background_edge_strength = None
                if np.any(boundary_magnitude):
                    boundary_edge_strength = round(float(boundary_magnitude.mean()), 3)
                else:
                    boundary_edge_strength = None

                foreground_edge_strength_list.append(foreground_edge_strength)
                background_edge_strength_list.append(background_edge_strength)
                boundary_edge_strength_list.append(boundary_edge_strength)
                crowdedness_list.append(0)

        # Iterates over all possible pairs of bboxes and computes the overlap for each of them
        for box_pair in list(itertools.combinations(range(len(bboxes)), 2)):
            # Calculates the corner coordinates for the intersection of the two bboxes
            x1 = max(bboxes[box_pair[0]][0], bboxes[box_pair[1]][0])
            y1 = max(bboxes[box_pair[0]][1], bboxes[box_pair[1]][1])
            x2 = min(bboxes[box_pair[0]][2], bboxes[box_pair[1]][2])
            y2 = min(bboxes[box_pair[0]][3], bboxes[box_pair[1]][3])
            # Calculates the area for the intersection
            inter_area = max(x2 - x1, 0) * max(y2 - y1, 0)
            if inter_area == 0:
                overlap1 = 0
                overlap2 = 0
            else:
                # Computes the overlap
                overlap1 = inter_area / bbox_area_list[box_pair[0]]
                overlap2 = inter_area / bbox_area_list[box_pair[1]]
                # Updates the ignore list, if one of the two bboxes is being overlapped by 60% or more (
                # heavily crowded)
                if overlap1 >= 0.6 or overlap2 >= 0.6:
                    # Determines which of the two bboxes has a higher distance
                    if distance_list[box_pair[0]] > distance_list[box_pair[1]]:
                        ignore_list[box_pair[0]] = 1  # instance with higher distance is ignored
                    else:
                        ignore_list[box_pair[1]] = 1

            # Uses the overlap values and scales them by the ratio between the two bboxes to compute the
            # crowdedness
            bboxes_ratio = min(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]]) / \
                           max(bbox_area_list[box_pair[0]], bbox_area_list[box_pair[1]])
            crowdedness_list[box_pair[0]] += overlap1 * bboxes_ratio
            crowdedness_list[box_pair[1]] += overlap2 * bboxes_ratio

        return instance_id_list, bboxes, label_list, bbox_area_list, ignore_list, bbox_height_list, \
            bbox_aspect_ratio_list, visible_instance_pixels_list, occlusion_ratio_list, distance_list, \
            foreground_brightness_list, contrast_to_background_list, entropy_list, background_edge_strength_list, \
            boundary_edge_strength_list, crowdedness_list

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
