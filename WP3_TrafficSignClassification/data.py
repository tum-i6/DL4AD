import os.path
import pickle

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, sampler
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch


class PickledDataset(Dataset):
    def __init__(self, file_path, transform):
        """
        Used to instantiate a pickled dataset object, which handles the information stored inside .pickle files.

        :param file_path: Path to the .pickle file.
        :param transform: The torch transforms instance, which should be applied to every image of the dataset for
        preprocessing.
        """
        # Loads the pickle file, which contains the dataset
        with open(file_path, mode='rb') as f:
            data = pickle.load(f)
            self.features = data['features']  # Image samples
            self.labels = data['labels']  # Labels
            self.count = len(self.labels)  # Amount of samples
            self.transform = transform  # Torch transform instance

    def __getitem__(self, index):
        """
        Extracts a specific sample from the dataset by its index value and the corresponding label.

        :param index: Index of the sample, which should be extracted.
        :return: The preprocessed image sample and the corresponding label.
        """
        feature = self.features[index]
        feature = self.transform(feature)  # Applies the torch transform (preprocessing) to the image sample
        return feature, self.labels[index]

    def __len__(self):
        """
        Returns the amount of samples from the dataset.

        :return: Amount of samples.
        """
        return self.count


class WrappedDataLoader:
    def __init__(self, dl, func):
        """
        Instantiates the WrappedDataloader instance, used for dynamically moving the data batches to the device.

        :param dl: Torch dataloader instance containing the dataset.
        :param func: Function, which should be dynamically applied to the data batches (move the data to the device).
        """
        self.dl = dl  # Dataloader instance
        self.func = func  # Dataloader function (move to device)

    def __len__(self):
        """
        Returns the amount of data from the dataloader.

        :return: Amount of data from the dataloader.
        """
        return len(self.dl)

    def __iter__(self):
        """
        Iterator for moving the dataloader batches to the device.
        """
        batches = iter(self.dl)
        for b in batches:
            yield self.func(*b)


def extend_dataset(dataset, num_classes):
    """
    Augments the GTSRB dataset with new samples using different data augmentation methods.

    :param dataset: A PickledDataset instance, which contains the image samples for GTSRB and the corresponding labels.
    :return: A PickledDataset instance with new attribute values, containing the new augmented dataset.
    """
    x = dataset.features  # Image samples
    y = dataset.labels  # Image labels
    #NUM_CLASSES = 15  # Amount of GTSRB classes

    # Placeholders for the augmented dataset image samples and labels
    x_extended = np.empty([0] + list(dataset.features.shape)[1:], dtype=dataset.features.dtype)
    y_extended = np.empty([0], dtype=dataset.labels.dtype)

    HORIZONTALLY_FLIPPABLE = [11, 12, 13, 15, 17, 18, 22, 26, 30, 35]  # Classes that can be horizontally flipped
    VERTICALLY_FLIPPABLE = [1, 5, 12, 15, 17]  # Classes that can be vertically flipped
    BOTH_FLIPPABLE = [32, 40]  # Classes that can be flipped in both directions
    CROSS_FLIPPABLE = np.array([  # Classes that are cross flippable
        [19, 20],
        [33, 34],
        [36, 37],
        [38, 39],
        [20, 19],
        [34, 33],
        [37, 36],
        [39, 38]
    ])

    # Iterates over all classes
    for c in tqdm(range(num_classes), desc='Augmenting each class'):
        x_extended = np.append(x_extended, x[y == c], axis=0)  # Adds the baseline samples to the augmented dataset

        if c in HORIZONTALLY_FLIPPABLE:
            x_extended = np.append(
                x_extended, x[y == c][:, :, ::-1, :], axis=0)  # Applies horizontal flipping to some baseline samples
        if c in VERTICALLY_FLIPPABLE:
            x_extended = np.append(
                x_extended, x[y == c][:, ::-1, :, :], axis=0)  # Applies vertical flipping to some baseline images
        if c in CROSS_FLIPPABLE[:, 0]:
            flip_c = CROSS_FLIPPABLE[CROSS_FLIPPABLE[:, 0] == c][0][1]
            x_extended = np.append(
                x_extended, x[y == flip_c][:, :, ::-1, :], axis=0)  # Applies cross flipping to some baseline images
        if c in BOTH_FLIPPABLE:
            x_extended = np.append(
                x_extended, x[y == c][:, ::-1, ::-1, :], axis=0)  # Applies both flippings to some baseline images

        y_extended = np.append(y_extended, np.full(  # Adds the corresponding amount of labels to the augmented dataset
            x_extended.shape[0]-y_extended.shape[0], c, dtype=y_extended.dtype))

    # Overwrites the dataset attributes of the PickledDataset instance with the new augmented dataset
    dataset.features = x_extended
    dataset.labels = y_extended
    dataset.count = len(y_extended)

    return dataset


def get_train_loaders(data_path, samples_per_class, batch_size, workers,num_classes, device):
    """
    Used to preprocess the train and validation data, which is afterwards being loaded into batches using a dataloader
    instance. The train data is being augmented during the preprocessing.

    :param data_path: Path to the data folder.
    :param samples_per_class: Amount of samples per class for the augmented train dataset.
    :param batch_size: Amount of samples, which are loaded per batch.
    :param workers: Amount of sub-processes to use for data loading.
    :param device: Device used for loading the data.
    :return: Dataloader instances for the train and validation dataset.
    """
    def to_device(x, y):
        """
        Used to load samples and labels onto the device.

        :param x: Image sample.
        :param y: Label.
        :return: Image sample and label, which are loaded to the device.
        """
        return x.to(device), y.to(device, dtype=torch.int64)

    # Loads the train and validation data
    print('Loading data from train.p')
    if os.path.exists(data_path + '/pickled_data/train_extended.p'):
        train_dataset =PickledDataset(data_path + '/pickled_data/train.p', transform=get_augmentation_transforms())
    else:

        train_dataset = extend_dataset(PickledDataset(  # Applies data augmentation to the train dataset
            data_path + '/pickled_data/train.p', transform=get_augmentation_transforms()), num_classes)
        with open(data_path + '/pickled_data/train_extended.p', 'wb') as pickle_file:
            pickle.dump(train_dataset,pickle_file)
    print('Loading data from valid.p')
    valid_dataset = PickledDataset(
        data_path + '/pickled_data/valid.p', transform=get_transforms())

    # Balances out the train dataset using a weight sampler
    class_sample_count = np.bincount(train_dataset.labels)  # Current amount of samples for every class
    # Weight for every class based on the amount of samples
    weights = 1 / np.array([class_sample_count[y] for y in train_dataset.labels])
    samp = sampler.WeightedRandomSampler(weights, num_classes * samples_per_class)  # Weight sampler instance

    print('Loading train data into the dataloader')
    train_loader = WrappedDataLoader(DataLoader(  # Loads the train data into the dataloader and the device
        train_dataset, batch_size=batch_size, sampler=samp, num_workers=workers), to_device)
    print('Loading validation data into the dataloader')
    valid_loader = WrappedDataLoader(DataLoader(  # Loads the validation data into the dataloader and the device
        valid_dataset, batch_size=batch_size, shuffle=False, num_workers=workers), to_device)
    print('Data loading finished! \n')

    return train_loader, valid_loader


def get_data_loader(data_path, dataset, batch_size, device):
    """
    Used to preprocess the data, which is afterwards being loaded into batches using a dataloader instance. Use this
    function for loading datasets that are used vor evaluation purposes.

    :param data_path: Path to the data folder.
    :param dataset: Specifies which dataset should be loaded (train, valid or test).
    :param batch_size: Amount of samples, which are loaded per batch.
    :param device: Device used for loading the data.
    :return: The dataloader for the dataset.
    """
    def to_device(x, y):
        """
        Used to load samples and labels onto the device.

        :param x: Image sample.
        :param y: Label.
        :return: Image sample and label, which are loaded to the device.
        """
        return x.to(device), y.to(device, dtype=torch.int64)

    # Loads the dataset and applies the transform to it
    print(f'Loading data from {dataset}.p')
    p_dataset = PickledDataset(f'{data_path}/pickled_data/{dataset}.p', transform=get_transforms())
    # Loads the transformed data into the dataloader
    print(f'Loading {dataset} data into the dataloader')
    data_loader = WrappedDataLoader(DataLoader(p_dataset, batch_size=batch_size, shuffle=False), to_device)
    print('Data loading finished! \n')

    return data_loader


def get_augmentation_transforms():
    """
    Used for train data augmentation and preprocessing. Normalizes and converts the data to torch tensors.

    :return: Transform instance for data augmentation and preprocessing.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([
            transforms.RandomRotation(20, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomAffine(0, translate=(0.2, 0.2), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomAffine(0, shear=20, interpolation=transforms.InterpolationMode.BICUBIC)
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.33999264, 0.31174879, 0.3209361], [0.27170149, 0.2598821, 0.26575037]),
    ])


def get_transforms():
    """
    Used for preprocessing the data and converting it to torch tensors.

    :return: Transform instance for preprocessing and torch conversion.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.33999264, 0.31174879, 0.3209361], [0.27170149, 0.2598821, 0.26575037]),
    ])


def denormalize_image(img):
    """
    Denormalizes an image, that was previously normalized.

    :param img: Torch tensor of a normalized image.
    :return: Torch tensor of an image without the normalization step.
    """
    img = img.detach()
    img[0, :, :] = img[0, :, :] * 0.27170149 + 0.33999264
    img[1, :, :] = img[1, :, :] * 0.2598821 + 0.31174879
    img[2, :, :] = img[2, :, :] * 0.26575037 + 0.3209361
    return img
