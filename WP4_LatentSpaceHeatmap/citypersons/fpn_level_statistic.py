import joblib
import torch
from torch import Tensor
import torchvision
import yaml
from matplotlib import patches, patheffects
from sklearn.metrics import log_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import numpy as np
import torchvision.transforms.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.poolers import LevelMapper

import utils
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import torchvision.models.detection.transform as T
import matplotlib.pyplot as plt
import data
import os
import pandas as pd
import tqdm
import cv2

from typing import Optional, List, Dict, Tuple, Union
from faster_rcnn_torchvision.contrastive_module import ContrastiveLoss
import faster_rcnn_torchvision
from faster_rcnn_torchvision.faster_rcnn import fasterrcnn_resnet50_fpn
from models import load_model
from evaluation import evaluate_model
from data import load_dataloaders


def show(imgs, id):
    if not isinstance(imgs, list):
        imgs = [imgs]
    # plt.figure(figsize=(30, 15))
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    # fig, axs = plt.subplots(nrows=len(imgs), squeeze=False, figsize=(12, 12))  # , figsize=(12, 24)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[i, 0].imshow(np.asarray(img))
        axs[i, 0].set_title('Input image with pred. Seg.')
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


def matching_det_gt(output, target, conf_thr=0.1, iou_thr=0.25):
    # print(f'Number of predictions {len()}')

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
        if target['ignore'][j] == 0:  # don't ignore
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
            dt_amount += 1  # After applying the score filter

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
                        img_eval_results[iou_threshold]['FNs'] -= 1  # cannot minus 2 more times the is_crowd Gts

        # Sets all IoU values for the matched prediction and gt bbox to 0, since they have been matched (except
        # for gt bboxes with the iscrowd property set to 1)
        if 'iscrowd' in gts[match[0]]:  # targeted to CityPersons dataset?
            if gts[match[0]]['iscrowd'] == 0:
                gts_dts_ious_conf_thr[match[0], :] = 0
        else:  # Targeted for KIA dataset
            gts_dts_ious_conf_thr[match[0], :] = 0
        gts_dts_ious_conf_thr[:, match[1]] = 0

        if match[0] not in matched_gts:
            matched_gts.append(match[0])  # Adds the gt_id to the list of matched ground truths

    return np.array(matched_gts)


def infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    assert possible_scales[0] == possible_scales[1]
    return possible_scales[0]


def level_mapping(resized_target, transformed_size, activations):
    scales = [infer_scale(feat, transformed_size) for feat in activations.values()]

    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = LevelMapper(lvl_min, lvl_max)([resized_target['boxes']])

    return map_levels


@torch.inference_mode()
def activation_plot(model, device, data_loader, transform):
    model.eval()
    max_dist = []

    for b, (images, targets) in tqdm.tqdm(enumerate(data_loader)):

        if 'masks' in targets[0]:  # instance seg
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore', 'masks')} for t in targets]
        else:  # object det
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore')} for t in targets]

        # size inference
        resized_images, resized_targets = transform(images, targets)

        for i, (img, target, resized_target) in enumerate(zip(images, targets, resized_targets)):
            resized_boxes = resized_target['boxes']
            resized_width = resized_boxes[:, 2] - resized_boxes[:, 0]  # x1 - x0
            resized_height = resized_boxes[:, 3] - resized_boxes[:, 1]  # y1 -y0

            resized_max_dist = torch.maximum(resized_height, resized_width)
            max_dist.append(resized_max_dist)

    result = torch.cat(max_dist, dim=0)
    fpn_lvl_0 = torch.sum(torch.where(torch.logical_and(0 <= result, result < 64), 1, 0))
    fpn_lvl_1 = torch.sum(torch.where(torch.logical_and(64 <= result, result < 128), 1, 0))
    fpn_lvl_2 = torch.sum(torch.where(torch.logical_and(128 <= result, result < 256), 1, 0))
    fpn_lvl_p6 = torch.sum(torch.where(torch.logical_and(256 <= result, result < 512), 1, 0))
    fpn_lvl_p7 = torch.sum(torch.where(torch.logical_and(512 <= result, result < float("inf")), 1, 0))

    fpn_lvls = ['fpn_lvl_0', 'fpn_lvl_1', 'fpn_lvl_2', 'fpn_lvl_p6', 'fpn_lvl_p7']
    counts = [fpn_lvl_0, fpn_lvl_1, fpn_lvl_2, fpn_lvl_p7, fpn_lvl_p7]
    # creating the bar plot
    plt.bar(fpn_lvls, counts)

    plt.xlabel("FPN levels")
    plt.ylabel("Counts of the associated instances")
    plt.title("FCOS with CityPersons dataset FPN level mapping")
    plt.show()
    plt.show()


def main():
    # get config file
    cfg_path = '.config/eval_config.yaml'

    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # draw
    transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    activation_plot(model, device, test_loader, transform)


if __name__ == '__main__':
    main()
