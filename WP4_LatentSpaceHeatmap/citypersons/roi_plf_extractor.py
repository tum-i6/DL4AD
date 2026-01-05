import joblib
import torch
import torchvision
import yaml
import numpy as np
import torchvision.transforms.functional as F

import utils
from torchvision.utils import draw_bounding_boxes
import torchvision.models.detection.transform as T
import matplotlib.pyplot as plt
import data
import os
import pandas as pd
import tqdm

from models import load_model
from data import load_dataloaders

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

daytime_type2num = {
    'day': 0,
    'medium': 1,
    'low': 2,
    None: float('nan')
}

sky_type2num = {
    'clear': 0,
    'low partly clouded': 1,
    'low completely covered': 2,
    None: float('nan')
}

wetness_type2num = {
    'dry': 0,
    'slightly moist': 1,
    'wet with puddles': 2,
    None: float('nan')
}

plfs_dict = {
    'CityPersons': [
        'image_id',
        'sample_id',
        'gt_boxes',
        'detection_result',
        'ignore',
        'crowd',
        'area',

        # plf values all numerical
        'bbox_height',
        'bbox_aspect_ratio',
        'visible_instance_pixels',
        'occlusion_ratio',
        'distance',
        'foreground_brightness',
        'contrast_to_background',
        'entropy',
        'background_edge_strength',
        'boundary_edge_strength',
        'crowdedness',

        # sample level plfs
        'brightness',
        'contrast',
        'edge_strength', ],

    'EuroCityPersons': [
        'image_id',
        'gt_boxes',
        'detection_result',
        'ignore',
        'crowd',
        'area',

        # plf values all numerical
        'bbox_height',
        'bbox_aspect_ratio',
        'entropy',
        'crowdedness',

        # sample level plfs
        'brightness',
        'contrast',
        'edge_strength', ],

    'KIA': [
        'image_id',
        'sample_id',
        'gt_boxes',
        'detection_result',
        'ignore',
        'crowd',
        'area',

        # plf values all numerical
        'bbox_height',
        'bbox_aspect_ratio',
        'visible_instance_pixels',
        'occlusion_ratio',
        'distance',
        'foreground_brightness',
        'contrast_to_background',
        'entropy',
        'background_edge_strength',
        'boundary_edge_strength',
        'crowdedness',

        # sample level plfs
        'brightness',
        'contrast',
        'edge_strength',

        # exclusive factors of KIA dataset
        'daytime_type',  # category
        'sky_type',  # category
        'wetness_type',  # category
        'fog_intensity',
        'vignette_intensity',
        'lens_flare_intensity']
}


def show(imgs, id):
    if not isinstance(imgs, list):
        imgs = [imgs]
    # plt.figure(figsize=(30, 15))
    fig, axs = plt.subplots(nrows=len(imgs), squeeze=False, figsize=(12, 12))  # , figsize=(12, 24)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[i, 0].axis('tight')
        axs[i, 0].axis('off')

    save_path = f'/path/to/output_dir/{id}_1000_outlier_scores.png'
    # plt.savefig(save_path, bbox_inches='tight')
    # plt.close(fig)
    plt.show()


def rgb(scores):
    scores = scores.detach().numpy()
    minimum, maximum = float(np.min(scores)), float(np.max(scores))
    heatmap_list = []
    for value in scores:
        ratio = 2 * (value - minimum) / (maximum - minimum)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        heatmap_list.append(tuple([r, g, b]))
    return heatmap_list


def matching_det_gt(output, target, conf_thr, iou_thr=0.25):
    """
    given the labels and the predictions of one input image, finally the indexes of the detected gt boxes will
    be returned as a numpy array.
    :param output: predictions
    :param target: ground truth
    :param conf_thr: predictions with lower confidence will be eliminated
    :param iou_thr: iou threshold used for the matching process
    :return: a numpy array containing the indexes of the detected gt boxes
    """
    # Extracts the ground truth information for each object
    gts = {}
    amount_safety_relevant_peds = 0  # Tracks the amount of pedestrians with the ignore property set to 0
    for j in range(len(target['boxes'])):
        gts[j] = {
            'bbox': target['boxes'][j],
            'area': target['area'][j],
            'ignore': target['ignore'][j]
        }
        # Extracts the information about the iscrowd property (if it exists)
        if 'iscrowd' in target:
            gts[j]['iscrowd'] = target['iscrowd'][j]
        if target['ignore'][j] == 0:
            amount_safety_relevant_peds += 1

    # Extracts the predictions for each detected object
    dts = {}
    if output['boxes'].nelement() != 0:
        for j in range(len(output['boxes'])):
            dts[j] = {
                'bbox': output['boxes'][j],
                # 'label': outputs[batch_idx]['labels'][j],
                'score': output['scores'][j]
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

    # Creates a copy of the gts_dts_iou matrix in order to filter it w.r. to the conf_thr
    gts_dts_ious_conf_thr = gts_dts_ious.copy()
    dt_amount = 0  # Counts the amount of detections after applying the conf_thr
    for dt_id in dts:
        if dts[dt_id]['score'].item() < float(conf_thr):
            # Resets the IoU values for detections with a confidence score below the threshold
            gts_dts_ious_conf_thr[:, dt_id] = 0
        else:
            dt_amount += 1

    img_eval_results = {
        iou_thr: {
            # Initially, all detections are considered as FPs and all safety relevant pedestrians are
            # considered as FNs
            'TPs': 0, 'SRTPs': 0, 'FPs': dt_amount, 'FNs': amount_safety_relevant_peds
        }}

    matched_gts = []  # A list used for tracking the gt_ids of matched bboxes
    # Matches the detections to the ground truth objects and fills the img_eval_results dictionary
    while True:
        if not np.any(gts_dts_ious_conf_thr):
            # Exit the loop if the matrix is an empty sequence
            break

        # Matrix coordinates of a ground truth and prediction bbox with the currently highest IoU
        match = np.unravel_index(gts_dts_ious_conf_thr.argmax(), gts_dts_ious_conf_thr.shape)

        if gts_dts_ious_conf_thr[match[0], match[1]] < iou_thr:
            # Exit the loop if the highest IoU value is lower than the lowest IoU threshold
            break

        # Computes the amount of TPs, FPs and FNs with respect to the IoU thresholds from img_eval_results
        for iou_threshold in [iou_thr]:
            if gts_dts_ious_conf_thr[match[0], match[1]] >= float(iou_threshold):
                img_eval_results[iou_threshold]['TPs'] += 1
                img_eval_results[iou_threshold]['FPs'] -= 1
                if gts[match[0]]['ignore'] == 0:
                    img_eval_results[iou_threshold]['SRTPs'] += 1
                    # This is a control sequence for safety relevant ground truths with the iscrowd property set
                    # to 1, since these can be matched multiple times
                    if match[0] not in matched_gts:
                        img_eval_results[iou_threshold]['FNs'] -= 1

        # Sets all IoU values for the matched prediction and gt bbox to 0, since they have been matched (except
        # for gt bboxes with the iscrowd property set to 1)
        if 'iscrowd' in gts[match[0]]:
            if gts[match[0]]['iscrowd'] == 0:
                gts_dts_ious_conf_thr[match[0], :] = 0
        else:
            gts_dts_ious_conf_thr[match[0], :] = 0
        gts_dts_ious_conf_thr[:, match[1]] = 0

        if match[0] not in matched_gts:
            matched_gts.append(match[0])  # Adds the gt_id to the list of matched ground truths

    return np.array(matched_gts)


@torch.inference_mode()
def extract_roi(model, device, data_loader, transform, feature_maps, roi_align, model_path, conf_thr=0.1, iou_thr=0.25,
                save_roi=False):
    model.eval()
    cpu_device = torch.device("cpu")

    # dataframe for saving
    df = pd.DataFrame(columns=[
        'image_id',
        'sample_id',
        'sample',
        'gt_boxes',
        'ignore',
        'crowd',
        'area',
        'roi',

        # plf values all numerical
        'bbox_height',
        'bbox_aspect_ratio',
        'visible_instance_pixels',
        'occlusion_ratio',
        'distance',
        'foreground_brightness',
        'contrast_to_background',
        'entropy',
        'background_edge_strength',
        'boundary_edge_strength',
        'crowdedness',

        # sample level plfs
        'brightness',
        'contrast',
        'edge_strength',

        # # exclusive factors of KIA dataset
        # 'daytime_type',  # category
        # 'sky_type',  # category
        # 'wetness_type',  # category
        # 'fog_intensity',
        # 'vignette_intensity',
        # 'lens_flare_intensity',
    ])

    for b, (images, targets) in enumerate(tqdm.tqdm(data_loader)):
        # if b == 5:
        #     break
        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        # Infers the batch into the model
        outputs = model(images)  # trigger the hooking

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # size inference
        resized_images, _ = transform(images)

        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):
            # matching predictions with ground truth
            matched_gts_index = matching_det_gt(output, target, conf_thr=conf_thr, iou_thr=iou_thr)

            gts_results = ['miss-detected: FN'] * len(target['boxes'])
            for index in matched_gts_index:
                gts_results[index] = 'detected: TP'

            # original ground truth boxes
            original_gts = target['boxes']
            # original input image
            original_h, original_w = img.size()[-2:]
            original_size = [original_h, original_w]

            # image size after transformation
            transformed_size = list(resized_images.image_sizes[i])

            # transformation ratios
            h_ratio, w_ratio = np.array(original_size) / np.array(transformed_size)
            # print(f' height ratio: {h_ratio:.3f}     weight ratio: {w_ratio:.3f}')

            # resize ground truth boxes
            resized_gts = T.resize_boxes(original_gts, original_size, transformed_size)

            # get feature maps
            activations = feature_maps.pop()

            # RoI align for the resized ground truth
            roi_align_outputs = roi_align(activations, [resized_gts.to(device)], resized_images.image_sizes)

            assert roi_align_outputs.shape[0] == resized_gts.shape[0]

            for j, roi in enumerate(roi_align_outputs.detach().cpu().numpy()):
                # pedestrian sample cropped
                x1, y1, x2, y2 = tuple(original_gts[j].cpu().numpy().astype(int))

                if (x2 - x1) * (y2 - y1) == 0:
                    continue

                sample = img[:, y1:y2, x1:x2].cpu().permute(1, 2, 0).numpy()

                # RoI feature
                if save_roi:
                    roi_feature = roi
                else:
                    roi_feature = float('nan')  # roi  # shape (C, H, W)

                # other info
                instance_id = target['instance id'][j]
                ignore = target['ignore'].cpu().numpy()[j]  # safety relevant pedestrians
                iscrowd = target['iscrowd'].cpu().numpy()[j]
                area = target['area'].cpu().numpy()[j]

                # plf values all numerical
                occ_ratio = target['occlusion_ratio'].cpu().numpy()[j]  # plf
                distance = target['distance'].cpu().numpy()[j]  # plf
                entropy = target['entropy'].cpu().numpy()[j]  # plf
                bbox_height = target['bbox_height'].cpu().numpy()[j]  # plf
                bbox_aspect_ratio = target['bbox_aspect_ratio'].cpu().numpy()[j]  # plf
                visible_instance_pixels = target['visible_instance_pixels'].cpu().numpy()[j]  # plf
                foreground_brightness = target['foreground_brightness'].cpu().numpy()[j]  # plf
                contrast_to_background = target['contrast_to_background'].cpu().numpy()[j]  # plf
                background_edge_strength = target['background_edge_strength'].cpu().numpy()[j]  # plf
                boundary_edge_strength = target['boundary_edge_strength'].cpu().numpy()[j]  # plf
                crowdedness = target['crowdedness'].cpu().numpy()[j]  # plf

                # pandas series for one sample
                s = pd.Series({
                    'image_id': int(b),
                    'sample_id': int(instance_id),
                    'sample': sample,
                    'gt_boxes': np.array([x1, y1, x2, y2]),
                    'detection_result': gts_results[j],
                    'ignore': ignore,
                    'crowd': iscrowd,
                    'area': area,
                    'roi': roi_feature,

                    # plf values all numerical
                    'bbox_height': bbox_height,
                    'bbox_aspect_ratio': bbox_aspect_ratio,
                    'visible_instance_pixels': visible_instance_pixels,
                    'occlusion_ratio': occ_ratio,
                    'distance': distance,
                    'foreground_brightness': foreground_brightness,
                    'contrast_to_background': contrast_to_background,
                    'entropy': entropy,
                    'background_edge_strength': background_edge_strength,
                    'boundary_edge_strength': boundary_edge_strength,
                    'crowdedness': crowdedness,

                    # sample level plfs
                    'brightness': target['brightness'],
                    'contrast': target['contrast'],
                    'edge_strength': target['edge_strength'],

                    # # exclusive factors of KIA dataset
                    # 'daytime_type': target['daytime_type'],  # category
                    # 'sky_type': target['sky_type'],  # category
                    # 'wetness_type': target['wetness_type'],  # category
                    # 'fog_intensity': target['fog_intensity'],
                    # 'vignette_intensity': target['vignette_intensity'],
                    # 'lens_flare_intensity': target['lens_flare_intensity'],
                })

                # append to dataframe
                df = df.append(s, ignore_index=True)

    # save dataframe as pickle
    parent_path = '/'.join(model_path.split('/')[:-1])
    saving_path = f'{parent_path}/evaluation_dataframe'
    if not os.path.isdir(saving_path):
        os.mkdir(saving_path)
    saving_path = f'{saving_path}/coco_test_set_roi_plfs_iou_{int(100 * float(iou_thr))}_conf_{int(100 * float(conf_thr))}.pkl'

    df.to_pickle(saving_path)


@torch.inference_mode()
def extract_plf(model, device, data_loader, model_path, conf_thr_list, saving_name=None, iou_thr=0.25):
    """
    extract the PLFs values for each instances of each images of the input data_loader, and finally a
    pandas dataframe will be saved.
    :param model: input model
    :param device: device
    :param data_loader: data loader
    :param model_path: the path of the model weight
    :param conf_thr_list: the list containing the various confidence thresholds
    :param saving_name: the name of saved dataframe
    :param iou_thr: the iou threshold used in the matching process
    """
    model.eval()
    cpu_device = torch.device("cpu")

    # infer the root path and the dataset name
    root_path = '/'.join(model_path.split('/')[:-1])
    dataset_name = root_path.split('/')[-1].split('_')[2]  # e.g. CityPersons

    assert dataset_name in ['CityPersons', 'EuroCityPersons', 'KIA']

    # placeholder dataframe for saving
    # note different datasets have different number of available PLFs
    df = pd.DataFrame(columns=plfs_dict[dataset_name])

    for b, (images, targets) in enumerate(tqdm.tqdm(data_loader)):  # batch level

        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        # Infers the batch into the model
        outputs = model(images)  # trigger the hooking

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):  # image level
            # free some space
            if 'masks' in target:
                del target['masks']
            if 'masks' in output:
                del output['masks']

            gts_results = {}  # a dict to store the detection results across different conf levels
            for conf_thr in conf_thr_list:
                # matching predictions with ground truth
                matched_gts_index = matching_det_gt(output, target, conf_thr=conf_thr, iou_thr=iou_thr)

                # detection result of each instance
                gts_results[conf_thr] = [0] * len(target['boxes'])  # 'miss-detected: FN' -> 0
                for index in matched_gts_index:
                    gts_results[conf_thr][index] = 1  # 'detected: TP' -> 1 due to string takes fewer memory

            for j, original_gt in enumerate(target['boxes']):  # instance level
                # pedestrian sample cropped
                x1, y1, x2, y2 = tuple(original_gt.cpu().numpy().astype(int))

                if (x2 - x1) * (y2 - y1) == 0:
                    continue

                # detection results across different conf_thre levels
                detection_results = [gts_results[key][j] for key in gts_results]
                # extract the PLFs values
                if dataset_name == 'CityPersons' or dataset_name == 'KIA':
                    # targets info
                    instance_id = target['instance id'][j]
                    ignore = target['ignore'].cpu().numpy()[j]  # safety relevant pedestrians
                    iscrowd = target['iscrowd'].cpu().numpy()[j]
                    area = target['area'].cpu().numpy()[j]

                    # plf values all numerical
                    occ_ratio = target['occlusion_ratio'].cpu().numpy()[j]  # plf
                    distance = target['distance'].cpu().numpy()[j]  # plf
                    entropy = target['entropy'].cpu().numpy()[j]  # plf
                    bbox_height = target['bbox_height'].cpu().numpy()[j]  # plf
                    bbox_aspect_ratio = target['bbox_aspect_ratio'].cpu().numpy()[j]  # plf
                    visible_instance_pixels = target['visible_instance_pixels'].cpu().numpy()[j]  # plf
                    foreground_brightness = target['foreground_brightness'].cpu().numpy()[j]  # plf
                    contrast_to_background = target['contrast_to_background'].cpu().numpy()[j]  # plf
                    background_edge_strength = target['background_edge_strength'].cpu().numpy()[j]  # plf
                    boundary_edge_strength = target['boundary_edge_strength'].cpu().numpy()[j]  # plf
                    crowdedness = target['crowdedness'].cpu().numpy()[j]  # plf

                elif dataset_name == 'EuroCityPersons':
                    ignore = target['ignore'].cpu().numpy()[j]  # safety relevant pedestrians
                    iscrowd = target['iscrowd'].cpu().numpy()[j]
                    area = target['area'].cpu().numpy()[j]

                    # plf values all numerical
                    entropy = target['entropy'].cpu().numpy()[j]  # plf
                    bbox_height = target['bbox_height'].cpu().numpy()[j]  # plf
                    bbox_aspect_ratio = target['bbox_aspect_ratio'].cpu().numpy()[j]  # plf
                    crowdedness = target['crowdedness'].cpu().numpy()[j]  # plf

                # pandas series for one sample
                if dataset_name == 'CityPersons':
                    s = pd.Series({
                        'image_id': int(b),
                        'sample_id': int(instance_id),
                        'gt_boxes': np.array([x1, y1, x2, y2]),

                        'detection_result': np.array(detection_results),
                        'ignore': ignore,
                        'crowd': iscrowd,
                        'area': area,

                        # plf values all numerical
                        'bbox_height': bbox_height,
                        'bbox_aspect_ratio': bbox_aspect_ratio,
                        'visible_instance_pixels': visible_instance_pixels,
                        'occlusion_ratio': occ_ratio,
                        'distance': distance,
                        'foreground_brightness': foreground_brightness,
                        'contrast_to_background': contrast_to_background,
                        'entropy': entropy,
                        'background_edge_strength': background_edge_strength,
                        'boundary_edge_strength': boundary_edge_strength,
                        'crowdedness': crowdedness,

                        # sample level plfs
                        'brightness': target['brightness'],
                        'contrast': target['contrast'],
                        'edge_strength': target['edge_strength']})

                elif dataset_name == "KIA":
                    s = pd.Series({
                        'image_id': int(b),
                        'sample_id': int(instance_id),
                        'gt_boxes': np.array([x1, y1, x2, y2]),

                        'detection_result': np.array(detection_results),
                        'ignore': ignore,
                        'crowd': iscrowd,
                        'area': area,

                        # plf values all numerical
                        'bbox_height': bbox_height,
                        'bbox_aspect_ratio': bbox_aspect_ratio,
                        'visible_instance_pixels': visible_instance_pixels,
                        'occlusion_ratio': occ_ratio,
                        'distance': distance,
                        'foreground_brightness': foreground_brightness,
                        'contrast_to_background': contrast_to_background,
                        'entropy': entropy,
                        'background_edge_strength': background_edge_strength,
                        'boundary_edge_strength': boundary_edge_strength,
                        'crowdedness': crowdedness,

                        # sample level plfs
                        'brightness': target['brightness'],
                        'contrast': target['contrast'],
                        'edge_strength': target['edge_strength'],

                        # exclusive factors of KIA dataset
                        'daytime_type': daytime_type2num[target['daytime_type']],  # category
                        'sky_type': sky_type2num[target['sky_type']],  # category
                        'wetness_type': wetness_type2num[target['wetness_type']],  # category
                        'fog_intensity': target['fog_intensity'],
                        'vignette_intensity': target['vignette_intensity'],
                        'lens_flare_intensity': target['lens_flare_intensity']})

                elif dataset_name == 'EuroCityPersons':
                    s = pd.Series({
                        'image_id': int(b),
                        'gt_boxes': np.array([x1, y1, x2, y2]),

                        'detection_result': np.array(detection_results),
                        'ignore': ignore,
                        'crowd': iscrowd,
                        'area': area,

                        # plf values all numerical
                        'bbox_height': bbox_height,
                        'bbox_aspect_ratio': bbox_aspect_ratio,
                        'entropy': entropy,
                        'crowdedness': crowdedness,

                        # sample level plfs
                        'brightness': target['brightness'],
                        'contrast': target['contrast'],
                        'edge_strength': target['edge_strength'],
                    })

                # append to dataframe
                df = df.append(s, ignore_index=True)

    # save dataframe as pickle
    parent_path = '/'.join(model_path.split('/')[:-1])
    saving_path = f'{parent_path}/evaluation_dataframe/PLFs'
    if not os.path.isdir(saving_path):
        os.makedirs(saving_path)
    saving_path = f'{saving_path}/{saving_name}_plfs_iou_{int(100 * float(iou_thr))}.pkl'
    # saving the dataframe to the specific path
    df.to_pickle(saving_path)


def draw_bboxes(model, device, data_loader, transform, rpn_boxes, roi_features):
    model.eval()
    cpu_device = torch.device("cpu")
    # load outlier detector
    filename = '/path/to/cityperson/IF_training_set_all_67_gap.sav'
    clf = joblib.load(filename)

    n = 0

    for b, (images, targets) in enumerate(data_loader):
        # if b == 3:
        #     exit()
        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        # Infers the batch into the model
        outputs = model(images)
        # rpn_boxes.pop()  # discard the final one -> redundent!
        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # roi features
        roi_feature = roi_features.pop()

        # size inference
        resized_images = transform(images)

        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):
            # exam if any occluded instance
            occlusion_ratio = target['occlusion_ratio']
            n_occlusion = torch.sum(occlusion_ratio >= 0.4).item()

            if n_occlusion >= 2:
                n += 1
                # if n == 3:
                #     exit()
                # original input image
                original_h, original_w = img.size()[-2:]
                original_size = [original_h, original_w]

                # resized images
                input_image_resized = resized_images[i]
                transformed_size = list(input_image_resized.image_sizes[0])

                # transformation ratios
                h_ratio, w_ratio = np.array(original_size) / np.array(transformed_size)
                # print(f'height ratio: {h_ratio:.3f}     weight ratio: {w_ratio:.3f}')

                # rpn output
                rpn_outputs = rpn_boxes[b][i].to(cpu_device)  # (1000, 4)
                # print(f'Number of outputs of rpn: {rpn_outputs.shape[0]}')

                # rpn transform transform
                rpn_outputs = T.resize_boxes(rpn_outputs, transformed_size, original_size)

                # outlier detections with roi features
                roi_feature = roi_feature.to(cpu_device)
                roi_feature = roi_feature.detach().numpy()
                roi_feature = np.apply_over_axes(np.mean, roi_feature, [2, 3])  # global average pooling
                roi_feature = roi_feature.reshape(roi_feature.shape[0], -1)

                pred = clf.predict(roi_feature)

                outlier_scores = clf.decision_function(
                    roi_feature)  # pos -> inlier; neg -> outlier; the lower, the more abnormal

                rpn_objectness = np.arange(1, len(rpn_outputs) + 1)[::-1]

                co_rela = np.corrcoef(outlier_scores, rpn_objectness)

                # print(f'Co-relation coe. between objectness and outlie scores is: {co_rela}')

                outlier_ratio = np.sum(pred != 1) / len(pred)

                # print(f'Outlier ratio out of {len(pred)} proposals: {round(outlier_ratio, 3)}')

                # draw the rpn boxes
                img_255 = img * 255
                # image with bounding boxes of RPN
                img_with_boxes = draw_bounding_boxes(img_255.to(torch.device("cpu"), torch.uint8),
                                                     boxes=torch.flip(rpn_outputs, dims=(0,)),  # rpn_outputs[::-1],
                                                     colors=rgb(torch.arange(len(rpn_outputs), )),
                                                     width=2)

                # image with outlier detection of bounding boxes of RPN
                sorted_outlier_scores, index = torch.sort(torch.tensor(outlier_scores))
                sorted_rpn_proposals = rpn_outputs[index]
                sorted_pred = pred[index]
                img_with_boxes_outlier_detector = draw_bounding_boxes(img_255.to(torch.device("cpu"), torch.uint8),
                                                                      boxes=sorted_rpn_proposals,  # rpn_outputs[::-1],
                                                                      colors=rgb(sorted_outlier_scores),
                                                                      # colors=[(255, 0, 0) if x == 1 else (0, 0, 255) for
                                                                      #         x in sorted_pred],
                                                                      width=2)

                # image with GT boxes
                img_with_gt = draw_bounding_boxes(img_255.to(torch.device("cpu"), torch.uint8),
                                                  boxes=target['boxes'],
                                                  colors=[(255, 0, 0) if x >= 0.4 else (0, 255, 0) for x in
                                                          occlusion_ratio.numpy()],
                                                  # colors=[(255, 0, 0) if x == 1 else (0, 255, 0) for x in
                                                  # crowd.numpy()],
                                                  width=2)

                # image with final detection bounding boxes
                img_with_dt = draw_bounding_boxes(img_255.to(torch.device("cpu"), torch.uint8),
                                                  boxes=output['boxes'],
                                                  colors=rgb(output['scores']),
                                                  width=2)

                show([img_with_gt, img_with_boxes, img_with_boxes_outlier_detector, img_with_dt], b)

            else:
                continue


def main():
    # get config file
    cfg_path = '/.config/eval_config.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # extract PLFs dataframe
    conf_threshold_list = np.arange(0, 101, 5) / 100
    extract_plf(
        model, device, test_loader, cfg['pretrained_weights_path'],
        conf_thr_list=conf_threshold_list,
        iou_thr=0.25,
    )


if __name__ == '__main__':
    main()
