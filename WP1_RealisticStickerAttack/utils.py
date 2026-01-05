from random import randrange
import os

import torchvision.transforms.functional as TF
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import cv2


# Model zoo (all supported models)
models = {'VGG16': models.vgg16_bn(),
          'VGG19': models.vgg19_bn(),
          'InceptionV3': models.inception_v3(init_weights=False),
          'ResNet18': models.resnet18(),
          'ResNet50': models.resnet50(),
          'ResNet101': models.resnet101(),
          'ResNet152': models.resnet152(),
          'WideResNet50': models.wide_resnet50_2(),
          'WideResNet101': models.wide_resnet101_2(),
          'ResNeXt50': models.resnext50_32x4d(),
          'ResNeXt101': models.resnext101_32x8d()}


def get_attack_algorithm_name(algorithm_abbreviation):
    """
    Returns the full name of the specified attack algorithm.

    :param algorithm_abbreviation: An abbreviation of the selected attack algorithm (Options: SSA, MCSA or RSA).
    :return: Saliency Sticker Application in case of SSA, Monte Carlo Sticker Application in case of MCSA and Random
    Sticker Application in case of RSA.
    """
    if algorithm_abbreviation == 'SSA':
        attack_algorithm = 'Saliency Sticker Application'
    elif algorithm_abbreviation == 'MCSA':
        attack_algorithm = 'Monte Carlo Sticker Application'
    elif algorithm_abbreviation == 'RSA':
        attack_algorithm = 'Random Sticker Application'
    else:
        raise ValueError(f'Your selection {algorithm_abbreviation} is not supported. Please use SSA, MCSA or RSA.')

    return attack_algorithm


def load_model(model_name, device_ids, weights_path):
    """
    Loads the classification model and the device.

    :param model_name: Name of the classification model, which should be loaded.
    :param device_ids: Device IDs for the GPUs, which should be used for training/evaluation.
    :param weights_path: Path to the model weights file.
    :return: The classification model and the device, which are being used for training/evaluation.
    """
    # Sets the device for training
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tqdm.write('\nDevice used for the attack: ' + str(device))
    tqdm.write('\nLoading ' + model_name + ' model (and weights).')

    # Loads the selected classification model and sets the amount of output parameters to 43 (amount of GTSRB classes)
    model = models[model_name]  # Instantiates the model

    # Sets the amount of output parameters for the VGG models
    if 'VGG' in model_name:
        model.classifier[6] = nn.Linear(4096, 43)
    # Sets the amount of output parameters for the InceptionV3 model
    elif model_name == 'InceptionV3':
        model.AuxLogits.fc = nn.Linear(768, 43)
        model.fc = nn.Linear(2048, 43)
    # Sets the amount of output parameters for the ResNet models (also includes ResNeXt) and adds a dropout layer
    elif 'Res' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_ftrs, 43)
        )

    model = nn.DataParallel(model)  # Sets the model to work in parallel on all available GPUs
    model.load_state_dict(torch.load(weights_path, map_location=device))  # Loads the model weights
    model = model.to(device)  # Moves the model to the device
    tqdm.write(model_name + ' has been loaded! \n')

    return model, device


def load_images(img_path, toTensor, device, color_format='RGB'):
    """
    Loads the images from the specified img_path and returns them as a list of torch tensors.

    :param img_path: Path to the image folder.
    :param toTensor: Torchvision transforms function used for torch conversion.
    :param device: The device used during the training.
    :param color_format: The color format for loading the images. Standard 'RGB'.
    :return: A list of torch tensors containing all the images from the img_path
    """
    torch_images = []  # List used as a placeholder for the images

    img_file_names = os.listdir(img_path)  # Loads the image file names from the data path

    for i in range(len(img_file_names)):
        img = Image.open(f"{img_path}/{img_file_names[i]}").convert(color_format)  # Loads the image using PIL
        torch_images.append(toTensor(img).to(device))  # Converts the image to torch and adds it to the output list

    return torch_images


def preprocess_images(img_list, device):
    """
    Preprocesses a list of torch images, right before model inference.

    :param img_list: List of torch tensor images.
    :param device: The device used during the training.
    :return: Torch tensor containing all preprocessed images.
    """
    # Placeholder for storing the final preprocessed torch images
    torch_data = torch.zeros((len(img_list), 3, img_list[0].shape[1], img_list[0].shape[2]))

    # Transforms instance used for normalization
    normalize = transforms.Normalize([0.33999264, 0.31174879, 0.3209361], [0.27170149, 0.2598821, 0.26575037])

    # Iterates over all images and applies the normalization
    for i in range(len(img_list)):
        torch_data[i] = normalize(img_list[i])

    return torch_data.to(device)


def get_accuracies(model_output, targets):
    """
    Calculates the prediction accuracy for the model predictions, based on the true (and attack) targets.

    :param model_output: Output torch tensor generated by the model.
    :param targets: A list of prediction targets, holding true (and attack) targets.
    :return: A list of accuracy values for the true (and attack) targets.
    """
    accuracies = [torch.sum(torch.argmax(model_output, 1) == targets[0]) / model_output.shape[0] * 100]

    if targets[1] is not None:
        accuracies.append(torch.sum(torch.argmax(model_output, 1) == targets[1]) / model_output.shape[0] * 100)

    return accuracies


def get_losses(criterion, model_output, targets):
    """
    Calculates the prediction loss for the model predictions, based on the true (and attack) targets.

    :param criterion: The loss function for calculating the prediction loss.
    :param model_output: Output torch tensor generated by the model.
    :param targets: A list of prediction targets, holding true (and attack) targets.
    :return: A list of loss values for the true (and attack) targets.
    """
    losses = [criterion(model_output, targets[0])]

    if targets[1] is not None:
        losses.append(criterion(model_output, targets[1]))

    return losses


def log_prediction_performance(accuracies, losses, display_text):
    """
    Logs the prediction performance to the console.

    :param accuracies: A dictionary of the following accuracies: train_accuracies and val_accuracies.
    :param losses: A dictionary of the following losses: train_losses and val_losses.
    :param display_text: A string, which should be displayed just before the prediction performance summary.
    """
    tqdm.write(display_text)
    tqdm.write('Accuracies:')
    tqdm.write(f"True target train accuracy:         {accuracies['train_accuracies'][0]:.2f}%")
    tqdm.write(f"True target validation accuracy:    {accuracies['val_accuracies'][0]:.2f}%")
    if len(accuracies['train_accuracies']) == 2:
        tqdm.write(f"Attack target train accuracy:       {accuracies['train_accuracies'][1]:.2f}%")
        tqdm.write(f"Attack target validation accuracy:  {accuracies['val_accuracies'][1]:.2f}%")
    tqdm.write('Losses:')
    tqdm.write(f"True target train loss:             {losses['train_losses'][0]:.7f}")
    tqdm.write(f"True target validation loss:        {losses['val_losses'][0]:.7f}")
    if len(losses['train_losses']) == 2:
        tqdm.write(f"Attack target train loss:           {losses['train_losses'][1]:.7f}")
        tqdm.write(f"Attack target validation loss:      {losses['val_losses'][1]:.7f}\n")


def get_prediction_performance(model, criterion, train_data, train_targets, val_data, val_targets, display_text):
    """
    Infers the train and validation data into the model to calculate the prediction accuracy and loss, based on the
    true and attack targets.

    :param model: The classification model, which is used for inference.
    :param criterion: Loss function for calculating the prediction loss.
    :param train_data: Preprocessed torch tensor containing the train images.
    :param train_targets: Tuple containing two torch tensors for the true and attack targets for the train images.
    :param val_data: Preprocessed torch tensor containing the validation images.
    :param val_targets: Tuple containing two torch tensors for the true and attack targets for the validation images.
    :param display_text: A string, which should be displayed to the console, just before the prediction performance
    summary.
    :return: Two dictionaries containing the accuracy and loss values for the train and validation data, based on the
    true and attack targets.
    """
    with torch.no_grad():
        accuracies = {}
        losses = {}

        # Infers the data into the model
        train_output = model(train_data)
        val_output = model(val_data)

        # Calculates the prediction accuracy and loss values
        accuracies['train_accuracies'] = get_accuracies(train_output, train_targets)
        accuracies['val_accuracies'] = get_accuracies(val_output, val_targets)
        losses['train_losses'] = get_losses(criterion, train_output, train_targets)
        losses['val_losses'] = get_losses(criterion, val_output, val_targets)

        # Logs the prediction performance to the console
        #log_prediction_performance(accuracies, losses, display_text)

        return accuracies, losses


def saliency2heatmap(saliency_tensor):
    """
    Converts the saliency map, which is given as a single channel torch tensor, into an RGB heatmap tensor by using
    a cv2 colormap transformation.

    :param saliency_tensor: Single channel tensor containing the saliency map.
    :return: RGB tensor containing the saliency heatmap.
    """
    toTensor = transforms.ToTensor()  # Converts numpy arrays to torch tensors

    heatmap = normalize_image(saliency_tensor)  # Normalizes the saliency map to contain values between 0 and 1
    heatmap = np.uint8(heatmap * 255)  # Converts the saliency map to numpy
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Converts the saliency map into a heatmap
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = toTensor(heatmap)  # Returns the saliency heatmap as a torch tensor

    return heatmap


def normalize_image(img):
    """
    Normalizes a torch tensor by subtracting its minimum value and dividing by the subtraction of the maximum and
    minimum value.

    :param img: Torch tensor of an image.
    :return: Torch tensor with normalized pixel values.
    """
    img = img.detach().to('cpu')
    img = (img - img.min()) / (img.max() - img.min())

    return img


def denormalize_image(img):
    """
    Denormalizes an image, that was previously preprocessed by the GTSRB normalization.

    :param img: Torch tensor of a preprocessed image.
    :return: Torch tensor of a preprocessed image without the normalization step.
    """
    img = img.detach().to('cpu')
    img[0, :, :] = img[0, :, :] * 0.27170149 + 0.33999264
    img[1, :, :] = img[1, :, :] * 0.2598821 + 0.31174879
    img[2, :, :] = img[2, :, :] * 0.26575037 + 0.3209361

    return img


def denormalize_images(images):
    """
    Iterates over all images and denormalizes them.

    :param images: Torch tensor of images, which should be denormalized.
    :return: A torch tensor of denormalized images.
    """
    denormalized_images = torch.zeros((images.shape[0], 3, images.shape[2], images.shape[3]))

    for i in range(images.shape[0]):
        denormalized_images[i] = denormalize_image(images[i])

    return denormalized_images


def get_saliency_map(pred_scores, data):
    """
    Sums up the prediction scores and backpropagates them to calculate the saliency maps for every image of the data.
    These saliency maps are used to calculate an average saliency map and to produce a visual heatmap.

    :param pred_scores: Prediction scores generated by the model for a specific class.
    :param data: The preprocessed torch tensor, which has been inferred into the model to generate the prediction
    scores.
    :return: An average saliency map, the same average saliency map converted into an RGB heatmap tensor, and a torch
    tensor containing all individual saliency maps ordered into a grid, based on the prediction scores.
    """
    pred_scores_sum = torch.sum(pred_scores)  # Sums up all the predictions scores
    pred_scores_sum.backward(retain_graph=True)  # Backpropagates on the summed prediction scores
    saliency_maps, _ = torch.max(data.grad.data.abs(), dim=1)  # Calculates the saliency maps for each image
    avg_saliency_map = torch.sum(saliency_maps, dim=0) / data.shape[0]  # Calculates an average saliency map
    heatmap = saliency2heatmap(avg_saliency_map.clone())  # Converts the average saliency map into a visual heatmap

    return avg_saliency_map, heatmap, saliency_maps


def get_saliency_maps(model, data, targets):
    """
    Infers the data into the model to generate class prediction scores. The prediction scores for the true target class,
    the classes with the highest prediction score and the attack target class, are summed up separately for all images
    to generate three unique prediction score values. Each of them is being backpropagated to yield the saliency maps
    of each image, based on the respective class. These saliency maps are used to calculate an average saliency map, and
    a visual heatmap.

    :param model: The classification model, which is used for inference.
    :param data: The preprocessed torch images, which are being inferred into the model.
    :param targets: Tuple containing two torch tensors for the true and attack targets for the given data.
    :return: A torch tensor containing an average saliency map for the true target class, a list of heatmaps (average
    true target heatmap, average highest prediction scores heatmap and average attack target heatmap) and a single torch
    tensor containing individual saliency heatmaps maps for each image based on three class scores, ordered into a grid.
    """
    heatmaps = []  # Placeholder for saving the heatmaps

    # Deactivates the gradient calculation for the model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Activates the gradient calculation for the data and infers it into the model
    if data.is_leaf:
        data.requires_grad = True
    else:
        data.retain_grad()
    output = model(data)

    # Calculates the saliency maps for the true target class
    true_target_pred_scores = output[:, targets[0][0]]
    true_target_avg_saliency, true_target_heatmap, true_target_saliencies = \
        get_saliency_map(true_target_pred_scores, data)
    heatmaps.append(true_target_heatmap)

    # Calculates the saliency maps for the predicted classes
    highest_pred_scores = torch.max(output, 1).values
    _, highest_pred_heatmap, highest_pred_saliencies = get_saliency_map(highest_pred_scores, data)
    heatmaps.append(highest_pred_heatmap)

    if targets[1] is not None:
        # Calculates the saliency maps for the attack target class
        attack_target_pred_scores = output[:, targets[1][0]]
        _, attack_target_heatmap, attack_target_saliencies = get_saliency_map(attack_target_pred_scores, data)
        heatmaps.append(attack_target_heatmap)

        # Generates a grid of individual saliency maps and the corresponding input images
        grid_saliency = torch.zeros((4 * data.shape[0], 3, data.shape[2], data.shape[3]))
        for i in range(data.shape[0]):
            grid_saliency[4*i] = denormalize_image(data[i].clone())
            grid_saliency[4*i + 1] = saliency2heatmap(true_target_saliencies[i])
            grid_saliency[4*i + 2] = saliency2heatmap(highest_pred_saliencies[i])
            grid_saliency[4*i + 3] = saliency2heatmap(attack_target_saliencies[i])
    else:
        # Generates a grid of individual saliency maps and the corresponding input images
        grid_saliency = torch.zeros((4 * data.shape[0], 3, data.shape[2], data.shape[3]))
        for i in range(data.shape[0]):
            grid_saliency[4*i] = denormalize_image(data[i].clone())
            grid_saliency[4*i + 1] = saliency2heatmap(true_target_saliencies[i])
            grid_saliency[4*i + 2] = saliency2heatmap(highest_pred_saliencies[i])
            grid_saliency[4*i + 3] = torch.zeros((3, data.shape[2], data.shape[3]))

    return true_target_avg_saliency, heatmaps, grid_saliency


def load_mask(mask_path, toTensor, device):
    """
    Loads the attack mask, used for constraining the sticker placement.

    :param mask_path: Path to the image file of the attack mask.
    :param toTensor: Torchvision transforms function used for torch conversion.
    :param device: The device used during the training.
    :return: A torch tensor of the attack mask, whose pixel values are either 0 or 1.
    """
    mask = Image.open(mask_path).convert('RGB')
    mask = torch.round(toTensor(mask))

    return mask.to(device)


def update_texture_iterator_postfix(texture_iterator, sticker_amount, accuracies, losses):
    """
    Updates the visual tqdm loading bar with the new prediction performance values.

    :param texture_iterator: The tqdm loading bar.
    :param sticker_amount: The amount of stickers, which are currently being applied.
    :param accuracies: A dictionary of the following accuracies: train_accuracies and val_accuracies.
    :param losses: A dictionary of the following losses: train_losses and val_losses.
    """
    if len(accuracies['train_accuracies']) == 2:
        texture_iterator.set_postfix_str(
            f"Top {sticker_amount}-Stickers --> "
            f"Acc:"
            f"[{accuracies['train_accuracies'][0]:.1f}%, "
            f"{accuracies['val_accuracies'][0]:.1f}%, "
            f"{accuracies['train_accuracies'][1]:.1f}%, "
            f"{accuracies['val_accuracies'][1]:.1f}%]"
            f" Loss:"
            f"[{losses['train_losses'][0]:.2f}, "
            f"{losses['val_losses'][0]:.2f}, "
            f"{losses['train_losses'][1]:.2f}, "
            f"{losses['val_losses'][1]:.2f}]")
    else:
        texture_iterator.set_postfix_str(
            f"Top {sticker_amount}-Stickers -->  "
            f"Train acc:{accuracies['train_accuracies'][0]:.1f}%, "
            f"Train loss:{losses['train_losses'][0]:.4f}, "
            f"Val acc:{accuracies['val_accuracies'][0]:.1f}%, "
            f"Val loss:{losses['val_losses'][0]:.4f}")


def random_resize(img, min_length, max_length, min_area, max_area, min_ratio, max_ratio):
    """
    Resizes a torch image randomly with respect to the size constraints.

    :param img: The torch tensor of the image, which should be resized.
    :param min_length: Minimum pixel length.
    :param max_length: Maximum pixel length.
    :param min_area: Minimum pixel area.
    :param max_area: Maximum pixel area.
    :param min_ratio: Minimum height/width pixel ratio.
    :param max_ratio: Maximum height/width pixel ratio.
    :return: The randomly resized torch image, with respect to the size constraints.
    """
    # Samples random height and width values until the size constraints are satisfied
    while True:
        height = randrange(min_length, max_length + 1)
        width = randrange(min_length, max_length + 1)
        area = height * width
        ratio = height / width

        if (max_area >= area >= min_area) and (max_ratio >= ratio >= min_ratio):
            # Resizes the image with respect to the sampled height and width value
            img = TF.resize(img, (height, width), antialias=True)

            return img


def random_rotate(img, max_rotation_angle):
    """
    Rotates a torch image randomly with respect to the maximum rotation angle.

    :param img: The torch tensor of the image, which should be rotated.
    :param max_rotation_angle: Maximum rotation angle.
    :return: The randomly rotated torch image, with respect to the maximum rotation angle.
    """
    # Rotates the sticker randomly with respect to the max_rotation_angle
    rotate = transforms.RandomRotation(max_rotation_angle, expand=True)

    return rotate(img)


def sticker2overlay(overlay, sticker, x, y):
    """
    Applies the RGBA sticker to the x and y pixel coordinates of the RGBA sticker overlay.

    :param overlay: An RGBA torch tensor, used for positioning the sticker.
    :param sticker: An RGBA torch tensor of the sticker that is being applied.
    :param x: The width pixel coordinate for positioning the sticker on the overlay.
    :param y: The height pixel coordinate for positioning the sticker on the overlay.
    :return: The RGBA overlay with the sticker applied to it.
    """
    # Calculates the height and width overlay position where the upper left corner of the sticker is gonna be positioned
    height_pos = y+sticker.shape[1] if overlay.shape[1] - (y+sticker.shape[1]) > 0 else overlay.shape[1]
    width_pos = x+sticker.shape[2] if overlay.shape[2] - (x+sticker.shape[2]) > 0 else overlay.shape[2]
    # Calculates the sticker height and width, with respect to the overlay boundaries
    sticker_height = sticker.shape[1] if overlay.shape[1] - (y+sticker.shape[1]) > 0 else height_pos - y
    sticker_width = sticker.shape[2] if overlay.shape[2] - (x+sticker.shape[2]) > 0 else width_pos - x

    # Applies the sticker to the overlay
    overlay[0:3, y:height_pos, x:width_pos] = \
        (sticker[0:3, :sticker_height, :sticker_width]*sticker[3, :sticker_height, :sticker_width]) + \
        (overlay[0:3, y:height_pos, x:width_pos]*(1-sticker[3, :sticker_height, :sticker_width]))

    # Updates the alpha channel values for the sticker overlay
    overlay[3, y:height_pos, x:width_pos] = \
        torch.clip(sticker[3, :sticker_height, :sticker_width] + overlay[3, y:height_pos, x:width_pos], 0, 1)

    return overlay


def check_sticker_mask_overlap(sticker, x, y, mask, overlay):
    """
    Checks whether the sticker positioned on the overlay with respect to the x and y coordinates overlaps the mask.
    Returns True if an overlap is detected, else returns False.

    :param sticker: An RGBA torch tensor of the sticker that is being applied.
    :param x: The width pixel coordinate for positioning the sticker on the overlay.
    :param y: The height pixel coordinate for positioning the sticker on the overlay.
    :param mask: Torch tensor of the attack mask, used for constraining the sticker placement.
    :param overlay: The torch tensor of the sticker overlay, used for sticker placement.
    :return: True if the sticker positioned on the overlay overlaps the mask. Else returns False.
    """
    # Transparent overlay used for positioning the sticker
    placement_overlay = overlay.clone()

    # Increases all pixel values of the sticker by 1 (for the red channel), in order to support the masked overlay
    # test for sticker placement check (avoiding pixel values of 0 for completely black stickers)
    enhanced_sticker = sticker.clone()
    enhanced_sticker[0] += (enhanced_sticker[3] != 0)

    # Applies the sticker to the placement overlay and multiplies it with the mask
    placement_overlay = sticker2overlay(placement_overlay, enhanced_sticker, x, y)
    masked_placement_overlay = torch.mul(placement_overlay[0:3], mask)

    # Calculates the pixel sum of the sticker overlay
    overlay_sum = torch.sum(placement_overlay[0:3])
    # Calculates the pixel sum of the masked sticker overlay
    masked_overlay_sum = torch.sum(masked_placement_overlay)

    # The position is valid if there is no intersection between the sticker position and the mask
    if (overlay_sum == masked_overlay_sum) and (overlay_sum != 0):
        return False
    else:
        return True


def get_random_position(sticker, mask, test_overlay):
    """
    Samples random height and width pixel coordinates to position the sticker with respect to the mask.

    :param sticker: Torch tensor of the sticker, which should be applied to the overlay.
    :param mask: Torch tensor of the attack mask, used for constraining the sticker placement.
    :param test_overlay: The torch tensor of the sticker overlay, used for sticker placement.
    :return: The width and height pixel coordinates for positioning the sticker.
    """
    # Samples random x, y coordinates for positioning the sticker until the position satisfies the mask
    while True:
        x = randrange(0, test_overlay.shape[2] - sticker.shape[2])
        y = randrange(0, test_overlay.shape[1] - sticker.shape[1])

        if not check_sticker_mask_overlap(sticker, x, y, mask, test_overlay):
            return x, y


def get_highest_saliency_position(sticker, saliency_map):
    """
    Calculates the width and height pixel coordinates of the strongest saliency point from the saliency map and uses
    them to calculate the sticker position coordinates, in order to cover the strongest saliency point.

    :param sticker: Torch tensor of the sticker, which should be applied to the overlay.
    :param saliency_map: A torch tensor of the prediction saliency map, which is used for covering the strongest points.
    :return: The width and height pixel coordinates for positioning the sticker.
    """
    # Calculates the x and y image coordinates for the strongest saliency point with respect to the sticker size
    index = torch.argmax(saliency_map)
    y = torch.div(index, saliency_map.shape[1], rounding_mode='floor') - \
        torch.div(sticker.shape[1], 2, rounding_mode='floor')
    y = torch.clip(y, 0, saliency_map.shape[1])
    x = torch.remainder(index, saliency_map.shape[2]) - torch.div(sticker.shape[2], 2, rounding_mode='floor')
    x = torch.clip(x, 0, saliency_map.shape[2])

    return x, y


def get_image_properties(images, device):
    """
    Iterates over all images and extracts the image brightness, contrast and blurriness values for each image.

    :param images: A list of torch images, whose properties should be extracted.
    :param device: The device used during the training.
    :return: A dictionary of brightness, contrast and blurriness values, where each image property is stored as a value
    inside a torch tensor.
    """
    # Used as a placeholder for storing the brightness, contrast and blurriness values for each image
    image_properties = {
        'brightness': torch.zeros(len(images)).to(device),
        'contrast': torch.zeros(len(images)).to(device),
        'blurriness': torch.zeros(len(images)).to(device)
    }

    for i in range(len(images)):
        # Calculates the brightness for the image
        image_properties['brightness'][i] = torch.mean(images[i]) / 0.5

        # Calculates the contrast for the image
        image_properties['contrast'][i] = torch.std(images[i]) / 0.26

        # Calculates the blurriness of the image
        image_properties['blurriness'][i] = \
            cv2.Laplacian((np.moveaxis(np.array(images[i].to('cpu')), 0, -1) * 255).astype(np.uint8), cv2.CV_8U).std()

    return image_properties


def sticker2images(victim_images, image_properties, sticker_overlay, overlay_mask, device):
    """
    Applies the sticker overlay to all victim images. Preprocesses the sticker overlay to fit each victim image
    individually by adjusting its brightness, contrast and motion blur.

    :param victim_images: A torch tensor of normalized torch images, which are being attacked by the sticker overlay.
    :param image_properties: A dictionary of brightness, contrast and blurriness values for each victim image.
    :param sticker_overlay: A torch tensor of the sticker overlay, used for the attack.
    :param overlay_mask: A torch tensor mask of non transparent areas from the sticker overlay.
    :param device: The device used during the training.
    :return: Victim images with the sticker overlay applied to them.
    """
    # Stacks the sticker overlay to the amount of victim images
    individual_sticker_overlays = sticker_overlay.repeat(victim_images.shape[0], 1, 1, 1)

    # Calculates the contrast of the sticker overlay to get the contrast ratio between background and foreground
    sticker_contrast = torch.std(sticker_overlay[0:3, overlay_mask]) / 0.26
    contrast_ratio = image_properties['contrast'] / sticker_contrast

    # Iterates over all sticker overlays from the stack and updates the brightness, contrast and blurriness values
    # with respect to the victim image properties
    for i in range(victim_images.shape[0]):
        # Adjusts the brightness
        individual_sticker_overlays[i, 0:3] = \
            TF.adjust_brightness(individual_sticker_overlays[i, 0:3], image_properties['brightness'][i])
        # Adjusts the contrast
        individual_sticker_overlays[i, 0:3] = \
            TF.adjust_contrast(individual_sticker_overlays[i, 0:3], contrast_ratio[i])
        # Applies the motion blur
        individual_sticker_overlays[i] = \
            apply_motion_blur(individual_sticker_overlays[i], image_properties['blurriness'][i], 13, device)

    # Normalizes the pixel values of all sticker overlays from the stack
    individual_sticker_overlays[:, 0] = (individual_sticker_overlays[:, 0] - 0.33999264) / 0.27170149
    individual_sticker_overlays[:, 1] = (individual_sticker_overlays[:, 1] - 0.31174879) / 0.2598821
    individual_sticker_overlays[:, 2] = (individual_sticker_overlays[:, 2] - 0.3209361) / 0.26575037

    # Applies the sticker stack to the victim images
    victim_images[:, 0] = \
        (individual_sticker_overlays[:, 3] * individual_sticker_overlays[:, 0]) + \
        ((1-individual_sticker_overlays[:, 3]) * victim_images[:, 0])
    victim_images[:, 1] = \
        (individual_sticker_overlays[:, 3] * individual_sticker_overlays[:, 1]) + \
        ((1-individual_sticker_overlays[:, 3]) * victim_images[:, 1])
    victim_images[:, 2] = \
        (individual_sticker_overlays[:, 3] * individual_sticker_overlays[:, 2]) + \
        ((1-individual_sticker_overlays[:, 3]) * victim_images[:, 2])

    return victim_images


def apply_motion_blur(image, blurriness_value, offset, device):
    """
    Applies motion blur to an image with respect to the specified offset and blurriness value.

    :param image: Torch tensor image, which should be blurred.
    :param blurriness_value: Specifies the blurriness strength.
    :param offset: The initial blurriness strength value.
    :param device: The device used during the training.
    :return: Torch tensor image with motion blur applied to it.
    """
    # Calculates the size of the motion blur kernel
    size = offset - int(torch.round(blurriness_value))
    if size % 2 == 0:
        size += 1

    # Generates the motion blur kernel
    kernel_motion_blur = torch.zeros((1, 1, size, size)).to(device)
    kernel_motion_blur[:, :, size // 2, :] = (1 / size)
    filter2D = nn.Conv2d(1, 1, kernel_size=size, bias=False, padding=size // 2, padding_mode='reflect').to(device)
    filter2D.weight.data = kernel_motion_blur
    filter2D.weight.requires_grad = False

    # Applies the motion blur to each channel of the image
    image[0] = torch.clip(filter2D(image[0].unsqueeze(0).unsqueeze(0)), 0, 1)
    image[1] = torch.clip(filter2D(image[1].unsqueeze(0).unsqueeze(0)), 0, 1)
    image[2] = torch.clip(filter2D(image[2].unsqueeze(0).unsqueeze(0)), 0, 1)
    image[3] = torch.clip(filter2D(image[3].unsqueeze(0).unsqueeze(0)), 0, 1)

    return image


def save_sticker_overlay(model_name, attack_algorithm, attack_target_class, destination_path, sticker_overlay,
                         sticker_amount, accuracies, sign_coverage):
    """
    Stores the sticker overlay as a .png image in RGB format. Stores all the attack information as the name of the
    image file.

    :param model_name: Name of the classification model, which is being attacked.
    :param attack_algorithm: Name of the attack algorithm used for the Realistic Sticker Attack.
    Options: RSA, MCSA or SSA.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param destination_path: Path to the output folder, where the image will be stored.
    :param sticker_overlay: A torch tensor of the sticker overlay, used for the attack, which should be stored.
    :param sticker_amount: The amount of stickers, which are applied to the sticker overlay.
    :param accuracies: A dictionary of 'train_accuracies' and 'val_accuracies' for the Realistic Sticker Attack using
    the sticker overlay.
    :param sign_coverage: The sign coverage of the stickers from the sticker overlay with respect to the sign surface
    area.
    """
    toPIL = transforms.ToPILImage()  # Used for converting images to PIL Images

    # Stores the best sticker overlay as a .png file
    file_name = f"{destination_path}/sticker_overlays/{model_name}_{attack_algorithm}_" \
                f"{sticker_amount}_sticker_attack_true_target_train_" \
                f"{accuracies['train_accuracies'][0]:.1f}%_val_" \
                f"{accuracies['val_accuracies'][0]:.1f}%"
    if attack_target_class in range(43):
        file_name += f"_attack_target_train_{accuracies['train_accuracies'][1]:.1f}%_val_" \
                     f"{accuracies['val_accuracies'][1]:.1f}%"
    file_name += f"_coverage_{sign_coverage:.1f}%.png"
    toPIL(sticker_overlay.to('cpu')).save(file_name)


def save_sticker_performance(performance_dict, sticker_amount, sign_coverage, attack_target_class, accuracies, losses):
    """
    Stores the prediction performance (accuracy and loss values) for the Realistic Sticker Attack inside a performance
    dictionary.

    :param performance_dict: A dictionary used for tracking the prediction performance for the Realistic Sticker Attack,
    using different amounts of stickers.
    :param sticker_amount: The amount of stickers, which are applied to the sticker overlay.
    :param sign_coverage: The sign coverage of the stickers from the sticker overlay with respect to the sign surface
    area.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param accuracies: A dictionary of 'train_accuracies' and 'val_accuracies' for the Realistic Sticker Attack using
    the sticker overlay.
    :param losses: A dictionary of 'train_losses' and 'val_losses' for the Realistic Sticker Attack using the sticker
    overlay.
    """
    # Stores the sticker performance for the current sticker overlay inside the performance dictionary
    performance_dict[f"{sticker_amount}-Sticker True Target Train Accuracy."] = accuracies['train_accuracies'][0]
    performance_dict[f"{sticker_amount}-Sticker True Target Val Accuracy"] = accuracies['val_accuracies'][0]
    performance_dict[f"{sticker_amount}-Sticker True Target Train Loss"] = losses['train_losses'][0]
    performance_dict[f"{sticker_amount}-Sticker True Target Val Loss"] = losses['val_losses'][0]
    performance_dict[f"{sticker_amount}-Sticker Sign Coverage"] = sign_coverage
    if attack_target_class in range(43):
        performance_dict[f"{sticker_amount}-Sticker Attack Target Train Accuracy"] = accuracies['train_accuracies'][1]
        performance_dict[f"{sticker_amount}-Sticker Attack Target Val Accuracy"] = accuracies['val_accuracies'][1]
        performance_dict[f"{sticker_amount}-Sticker Attack Target Train Loss"] = losses['train_losses'][1]
        performance_dict[f"{sticker_amount}-Sticker Attack Target Val Loss"] = losses['val_losses'][1]


def visualize_attack2TensorBoard(
        writer, sticker_amount, attack_target_class, device, sticker_overlay, mask, clean_avg_train_saliency,
        ssa_saliency, attacked_train_heatmaps, attacked_val_heatmaps,
        attacked_train_individual_heatmaps, attacked_val_individual_heatmaps
):
    """
    Visualizes the Realistic Sticker Attack in TensorBoard by storing the sticker overlay, the attack mask, the saliency
    maps and the heatmaps for the model predictions.

    :param writer: A TensorBoard SummaryWriter instance.
    :param sticker_amount: The amount of stickers, which are applied to the sticker overlay.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param device: The device used during the training.
    :param sticker_overlay: A torch tensor of the sticker overlay, used for the attack.
    :param mask: Torch tensor of the attack mask, used for constraining the sticker placement.
    :param clean_avg_train_saliency: A torch tensor of an average saliency heatmap map for the clean train images.
    :param ssa_saliency: A torch tensor of an average raw saliency map (grayscale) for the clean train images.
    :param attacked_train_heatmaps: A torch tensor of an average saliency heatmap for the attacked train images.
    :param attacked_val_heatmaps: A torch tensor of an average saliency heatmap for the attacked val images.
    :param attacked_train_individual_heatmaps: A torch tensor of individual saliency heatmaps for the attacked train
    images.
    :param attacked_val_individual_heatmaps: A torch tensor of individual saliency heatmaps for the attacked val images.
    """
    # Visualizes the RGBA sticker overlay
    writer.add_image(f"Training {sticker_amount}-Sticker Attack / 1. Transparent Sticker Overlay", sticker_overlay)
    # Visualizes the attack mask, the RGB sticker overlay, the raw true target saliency map and the covered saliency map
    writer.add_images(
        f"Training {sticker_amount}-Sticker Attack / 2. Attack Mask, Sticker Overlay, "
        f"Raw Average True Target Train Saliency Map, Covered Saliency Map",
        torch.stack((
            mask,
            sticker_overlay[0:3],
            normalize_image(torch.mul(clean_avg_train_saliency,
                                      torch.ones((3, mask.shape[1], mask.shape[2])).to(device))).to(device),
            normalize_image(ssa_saliency).to(device)
        )))

    if attack_target_class in range(43):
        # Visualizes the average train and validation saliency heatmaps (including the attack target class)
        writer.add_images(
            f"Training {sticker_amount}-Sticker Attack / 3. Average Train & Val Saliency Heatmaps "
            f"(True Target, Average Prediction, Attack Target)",
            torch.stack((
                attacked_train_heatmaps[0],
                attacked_train_heatmaps[1],
                attacked_train_heatmaps[2],
                torch.zeros((3, mask.shape[1], mask.shape[2])),
                attacked_val_heatmaps[0],
                attacked_val_heatmaps[1],
                attacked_val_heatmaps[2]
            )))
    else:
        # Visualizes the average train and validation saliency heatmaps
        writer.add_images(
            f"Training {sticker_amount}-Sticker Attack / 3. Average Train & Val Saliency Heatmaps "
            f"(True Target & Average Prediction)",
            torch.stack((
                attacked_train_heatmaps[0],
                attacked_train_heatmaps[1],
                torch.zeros((3, mask.shape[1], mask.shape[2])),
                attacked_val_heatmaps[0],
                attacked_val_heatmaps[1]
            )))

    # Visualizes the individual saliency heatmaps for the train and validation data
    writer.add_images(
        f"Training {sticker_amount}-Sticker Attack / 4. Train Input Images, True Target Saliency Maps, "
        f"Average Prediction Saliency Maps (& Attack Target Saliency Maps)", attacked_train_individual_heatmaps)
    writer.add_images(
        f"Training {sticker_amount}-Sticker Attack / 4. Val Input Images, True Target Saliency Maps, "
        f"Average Prediction Saliency Maps (& Attack Target Saliency Maps)", attacked_val_individual_heatmaps)

    writer.close()


def scalars_attack2TensorBoard(
        writer_train, writer_val, sticker_amount, attack_target_class, sign_coverage, accuracies, losses
):
    """
    Logs the prediction performance (accuracy and loss value) for the Realistic Sticker Attack to TensorBoard.

    :param writer_train: A TensorBoard SummaryWriter instance used for train data.
    :param writer_val:  A TensorBoard SummaryWriter instance used for validation data.
    :param sticker_amount: The amount of stickers, which are applied to the sticker overlay.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param sign_coverage: The sign coverage of the stickers from the sticker overlay with respect to the sign surface
    area.
    :param accuracies: A dictionary of 'train_accuracies' and 'val_accuracies' for the Realistic Sticker Attack using
    the sticker overlay.
    :param losses: A dictionary of 'train_losses' and 'val_losses' for the Realistic Sticker Attack using the sticker
    overlay.
    """
    # Logs the prediction performance for the current sticker overlay to TensorBoard
    writer_train.add_scalar('Accuracy/True Target Accuracy (in %)', accuracies['train_accuracies'][0], sticker_amount)
    writer_val.add_scalar('Accuracy/True Target Accuracy (in %)', accuracies['val_accuracies'][0], sticker_amount)
    writer_train.add_scalar('Loss/True Target Loss', losses['train_losses'][0], sticker_amount)
    writer_val.add_scalar('Loss/True Target Loss', losses['val_losses'][0], sticker_amount)
    writer_val.add_scalar('Sign Coverage/Sign Coverage (in %)', sign_coverage, sticker_amount)
    if attack_target_class in range(43):
        writer_train.add_scalar('Accuracy/Attack Target Accuracy (in %)',
                                accuracies['train_accuracies'][1], sticker_amount)
        writer_val.add_scalar('Accuracy/Attack Target Accuracy (in %)', accuracies['val_accuracies'][1], sticker_amount)
        writer_train.add_scalar('Loss/Attack Target Loss', losses['train_losses'][1], sticker_amount)
        writer_val.add_scalar('Loss/Attack Target Loss', losses['val_losses'][1], sticker_amount)
    writer_train.close()
    writer_val.close()


def scalars_baseline2TensorBoard(writer, sticker_amount, attack_target_class, accuracies, losses):
    """
    Logs the baseline prediction performance on clean images to TensorBoard.

    :param writer: A TensorBoard SummaryWriter instance.
    :param sticker_amount: The amount of stickers, which are applied to the sticker overlay.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param accuracies: A dictionary of 'train_accuracies' and 'val_accuracies' for the clean prediction performance.
    :param losses: A dictionary of 'train_losses' and 'val_losses' for the clean prediction performance.
    """
    # Logs the baseline prediction performance on clean images to TensorBoard
    writer.add_scalar('Accuracy/True Target Accuracy (in %)', accuracies['train_accuracies'][0], sticker_amount)
    writer.add_scalar('Accuracy/True Target Accuracy (in %)', accuracies['val_accuracies'][0], sticker_amount)
    writer.add_scalar('Loss/True Target Loss', losses['train_losses'][0], sticker_amount)
    writer.add_scalar('Loss/True Target Loss', losses['val_losses'][0], sticker_amount)
    if attack_target_class in range(43):
        writer.add_scalar('Accuracy/Attack Target Accuracy (in %)', accuracies['train_accuracies'][1], sticker_amount)
        writer.add_scalar('Accuracy/Attack Target Accuracy (in %)', accuracies['val_accuracies'][1], sticker_amount)
        writer.add_scalar('Loss/Attack Target Loss', losses['train_losses'][1], sticker_amount)
        writer.add_scalar('Loss/Attack Target Loss', losses['val_losses'][1], sticker_amount)
    writer.close()


def visualize_baseline2TensorBoard(
        writer, attack_target_class, device, mask, clean_avg_train_saliency, clean_train_heatmaps, clean_val_heatmaps,
        clean_train_individual_heatmaps, clean_val_individual_heatmaps
):
    """
    Visualizes the baseline Realistic Sticker Attack in TensorBoard by storing the attack mask, the average train
    saliency map, the average train and validation saliency heatmaps and the individual train and validation saliency
    heatmaps.

    :param writer: A TensorBoard SummaryWriter instance.
    :param attack_target_class: The class which should be predicted. The target for the attack. Set to None if only
    misclassification is to be achieved.
    :param device: The device used during the training.
    :param mask: Torch tensor of the attack mask, used for constraining the sticker placement.
    :param clean_avg_train_saliency: A torch tensor of an average saliency heatmap map for the clean train images.
    :param clean_train_heatmaps: A torch tensor of an average saliency heatmap for the clean train images.
    :param clean_val_heatmaps: A torch tensor of an average saliency heatmap for the clean val images.
    :param clean_train_individual_heatmaps: A torch tensor of individual saliency heatmaps for the clean train images.
    :param clean_val_individual_heatmaps: A torch tensor of individual saliency heatmaps for the clean val images.
    """
    # Visualizes the attack mask and the raw average true target train saliency map (grayscale)
    writer.add_images(
        f"Baseline / 1. Attack Mask & Raw Average True Target Train Saliency Map",
        torch.stack((
            mask,
            normalize_image(torch.mul(clean_avg_train_saliency,
                                      torch.ones((3, mask.shape[1], mask.shape[2])).to(device))).to(device)
        )))
    if attack_target_class in range(43):
        # Visualizes the average train and validation saliency heatmaps (including the attack target class)
        writer.add_images(
            f"Baseline / 2. Average Train & Val Saliency Heatmaps (True Target, Average Prediction, Attack Target)",
            torch.stack((clean_train_heatmaps[0], clean_train_heatmaps[1], clean_train_heatmaps[2],
                         torch.zeros((3, mask.shape[1], mask.shape[2])), clean_val_heatmaps[0], clean_val_heatmaps[1],
                         clean_val_heatmaps[2]))
        )
    else:
        # Visualizes the average train and validation saliency heatmaps
        writer.add_images(f"Baseline / 2. Average Train & Val Saliency Heatmaps (True Target & Average Prediction)",
            torch.stack((clean_train_heatmaps[0], clean_train_heatmaps[1],
                         torch.zeros((3, mask.shape[1], mask.shape[2])),
                         clean_val_heatmaps[0], clean_val_heatmaps[1])))

    # Visualizes the individual saliency heatmaps for the train and validation data
    writer.add_images(
        f"Baseline / Train Input Images, True Target Saliency Maps, "
        f"Average Prediction Saliency Maps (& Attack Target Saliency Maps)", clean_train_individual_heatmaps)
    writer.add_images(
        f"Baseline / Val Input Images, True Target Saliency Maps, "
        f"Average Prediction Saliency Maps (& Attack Target Saliency Maps)", clean_val_individual_heatmaps)
    writer.close()
