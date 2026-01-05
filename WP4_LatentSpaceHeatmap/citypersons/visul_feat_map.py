from collections import OrderedDict

import tqdm
from random import random

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
from torchvision.ops.poolers import LevelMapper
from torchvision.utils import draw_segmentation_masks
import matplotlib.pyplot as plt
import cv2

from typing import Optional, List, Dict, Tuple, Union
from models import load_model
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

    save_path = f'/path/to/{id}_1000_outlier_scores.png'
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


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Plots one bounding box x on image img with the specific color, line thickness and label (optional)
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


@torch.inference_mode()
def visul_heatmap(model, device, data_loader, feature_maps, tag, saving_root, conf_thres=0.5, n_imgs=10):
    """
    save the input image with detections, input image overlaying heatmap and optional input image
    overlaying the predicted binary mask. Only the first n_imgs images of the data_loader will be saved.
    :param model: trained model
    :param device: device for inference
    :param data_loader: input data_loader
    :param feature_maps: the extracted feature maps
    :param tag: tag for the name of the saved images
    :param saving_root: the root of saving
    :param conf_thres: predictions with lower confidence than the threshold will be eliminated from the visualization
    :param n_imgs: the number of images to be saved
    """
    model.eval()
    cpu_device = torch.device("cpu")
    # iterate the batches in the input data_loader
    for b, (images, targets) in enumerate(data_loader):
        if b > n_imgs:
            break

        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        if 'masks' in targets[0]:  # instance seg
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore', 'masks')} for t in targets]
        else:  # object det
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore')} for t in targets]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets are a list of dicts

        # Infers the batch into the model
        outputs = model(images)

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # eliminate the predictions lower than the pre-defined conf thres
        outputs = [{k: v[t['scores'] >= conf_thres] for k, v in t.items()} for t in outputs]

        activations = feature_maps.pop()

        activations = {k: v.detach().cpu() for k, v in activations.items()}
        # integrate the images in the current batch
        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):
            # input images
            img_255_tensor = img * 255
            img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
            img_255 = img_255_tensor.numpy().transpose(1, 2, 0)  # CHW to HWC

            # 0. input image with gts
            gt_bboxes = target['boxes'].clone()
            img_gt = cv2.cvtColor(np.copy(img_255), cv2.COLOR_RGB2BGR)  # RGB to BGR
            for bbox in gt_bboxes:
                img_gt = plot_one_box(bbox, img_gt, color=[197, 245, 136], line_thickness=1)

            # 1. input image with dets
            det_bboxes = output['boxes'].clone()
            det_scores = output['scores'].clone()
            img_det = cv2.cvtColor(np.copy(img_255), cv2.COLOR_RGB2BGR)  # RGB to BGR

            for bbox, conf in zip(det_bboxes, det_scores):
                label = f'{conf:.2f}'
                img_det = plot_one_box(bbox, img_det, color=[0, 47, 167], label=label, line_thickness=1)

            i_h, i_w = img.size()[-2], img.size()[-1]  # size of input image

            for k, v in activations.items():
                # only the feature maps of the 1st FPN layer are needed
                if k != '0':
                    continue

                # create masks for each activation levels
                mask = torch.zeros_like(img.cpu())
                for gt in target['boxes'].clone():
                    x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
                    mask[:, y1: y2, x1: x2] = 1
                mask = mask.numpy().transpose(1, 2, 0)

                # # channel-wise average the activations
                heatmap_avg = torch.mean(v, 1).squeeze(0).numpy()
                heatmap = (heatmap_avg - np.min(heatmap_avg)) / (
                        np.max(heatmap_avg) - np.min(heatmap_avg))  # (H, W) 0-1

                # heatmap = 1 - heatmap  # tricky TODO: put this into a func

                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (i_w, i_h))

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

                # overlaying heatmap and resized input image.
                feat_overlay = (heatmap * 0.5 + img_255 * 0.5)  # RGB
                feat_overlay = cv2.cvtColor(feat_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR

                # 3. mask
                # plot the heatmap mask
                heatmap_mask = heatmap_avg
                heatmap_mask = cv2.resize(heatmap_mask, (i_w, i_h))
                heatmap_mask = 1 / (1 + np.exp(-heatmap_mask)) > .5  # sigmoid func

                copy = np.copy(img_255)
                copy[heatmap_mask == 0] = 255
                # overlay the input image with the predicted binary mask
                mask_overlay = (0.75 * copy + 0.25 * img_255)
                mask_overlay = cv2.cvtColor(mask_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR
                # draw the detection boxes in the overlays
                for bbox, conf in zip(det_bboxes, det_scores):
                    label = f'{conf:.2f}'
                    mask_overlay = plot_one_box(bbox, mask_overlay, color=[0, 47, 167], label=label, line_thickness=1)

            # save images
            cv2.imwrite(f'{saving_root}/{tag}_image_{b}_labels.png', img_gt)
            cv2.imwrite(f'{saving_root}/{tag}_image_{b}_dets.png', img_det)
            cv2.imwrite(f'{saving_root}/{tag}_image_{b}_heatmap.png', feat_overlay)
            cv2.imwrite(f'{saving_root}/{tag}_image_{b}_mask_dets.png', mask_overlay)


@torch.inference_mode()
def activation_plot(model, device, data_loader, transform, feature_maps):
    model.eval()
    cpu_device = torch.device("cpu")

    for b, (images, targets) in enumerate(data_loader):
        if b > 1:
            exit()

        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        if 'masks' in targets[0]:  # instance seg
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore', 'masks')} for t in targets]
        else:  # object det
            targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore')} for t in targets]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets are a list of dicts

        # Infers the batch into the model
        outputs = model(images)

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        # size inference
        resized_images, resized_targets = transform(images, targets)

        for i, (img, target, resized_target, output) in enumerate(zip(images, targets, resized_targets, outputs)):
            # original input image
            original_h, original_w = img.size()[-2:]
            original_size = [original_h, original_w]
            print(f'The original input size is {original_size}')

            # image size after transformation
            transformed_size = list(resized_images.image_sizes[i])
            print(f'The input size after transformation is {transformed_size}')

            # transformation ratios
            h_ratio, w_ratio = np.array(original_size) / np.array(transformed_size)

            print(h_ratio, w_ratio)

            # get feature maps
            # 0: 256*200*304; 1: 256*100*152; 2: 256*50*76; 3: 256*25*38; pool: 256*13*19
            # PLUS: No relu performed on these feature maps
            activations = feature_maps.pop()
            if isinstance(activations, torch.Tensor):
                activations = OrderedDict([("0", activations)])
            # del activations['pool']

            activations = {k: v.detach().cpu() for k, v in activations.items()}
            for k, v in activations.items():
                print(f'FPN level {k}')
                print(f'shape {v.shape}')

            # map_levels = level_mapping(resized_target, transformed_size, activations)

            result_list = []
            name_list = []
            level_mask = {}

            bce_loss = 0

            # plot the input images
            img_255_tensor = img * 255
            img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
            img_255 = img_255_tensor.numpy().transpose(1, 2, 0)

            # get the GT bboxes coordinates from the target
            tar_bboxes = target['boxes'].clone().cpu()
            tar_bboxes[:, [2]] = tar_bboxes[:, [2]] - tar_bboxes[:, [0]]
            tar_bboxes[:, [3]] = tar_bboxes[:, [3]] - tar_bboxes[:, [1]]

            # get the det bboxes coordinates from the target
            det_bboxes = output['boxes'].clone()
            det_bboxes[:, [2]] = det_bboxes[:, [2]] - det_bboxes[:, [0]]
            det_bboxes[:, [3]] = det_bboxes[:, [3]] - det_bboxes[:, [1]]
            det_scores = output['scores'].clone()

            # Plot input images and Gt annotations
            fig = plt.figure()
            ax2 = fig.add_subplot()
            plt.imshow(img_255)
            if 'masks' in output:
                bool_mask = target['masks'].clone().cpu() > 0.5
                bool_mask = bool_mask.squeeze(1)
                img_mask = draw_segmentation_masks(img_255_tensor, bool_mask, alpha=0.9)
                plt.imshow(np.asarray(F.to_pil_image(img_mask.detach())))
            plt.axis('off')
            # add boxes
            for idx, box in enumerate(tar_bboxes):
                rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='g',
                                         facecolor='none')
                # ax.add_patch(rect)
                ax2.set_title('Input image with Ground Truth')
                ax2.add_patch(rect)

            plt.tight_layout()
            plt.show()  # input image with Gts

            for k, v in activations.items():
                if k != '0':
                    continue
                print(f'layer {k} with feature map shape {v.shape}')
                # create masks for each activation levels
                mask = torch.zeros_like(img.cpu())
                for gt in target['boxes'].clone():
                    x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
                    mask[:, y1: y2, x1: x2] = 1
                mask_tensor = mask
                mask = mask.numpy().transpose(1, 2, 0)
                # level_mask[int(k)] = mask

                # plot the binary mask
                plt.imshow(mask, cmap='gray')
                plt.axis('off')
                plt.tight_layout()
                plt.show()  # binary mask

                # # plot human semantic segmentation mask
                # bool_mask = target['masks'].clone().cpu()
                # bool_mask = bool_mask.squeeze(1)  # N, H, W
                # bool_mask_per_instance = bool_mask
                # bool_mask = torch.mean(torch.tensor(bool_mask, dtype=torch.float), dim=0)  # 1, H, W
                # if len(torch.unique(bool_mask)) != 2:
                #     print(len(torch.unique(bool_mask)))
                #     for x in bool_mask_per_instance:
                #         plt.imshow(x)
                #         plt.axis('off')
                #         plt.tight_layout()
                #         plt.show()
                #     # bool_mask = torch.where(bool_mask == 0, .1, .9)
                #     plt.imshow(bool_mask)
                #     plt.axis('off')
                #     plt.tight_layout()
                #     plt.show()

                # name_list.append(f'FPN layer{int(k) + 1}')

                # plot the heatmap
                f_h, f_w = v.shape[-2:]
                i_h, i_w = tuple(original_size)

                # for idx, heatmap in enumerate(v.squeeze(0).numpy()):  # C H W
                #     if idx == 10:
                #         break
                #     # heatmap from layers with (H, W) 0-1
                #     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # (H, W) 0-1
                #
                #     heatmap = np.uint8(255 * heatmap)
                #     heatmap = cv2.resize(heatmap, (i_w, i_h))
                #
                #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                #
                #     b, g, r = cv2.split(heatmap)
                #     heatmap = cv2.merge([r, g, b])
                #     result_list.append(heatmap)
                #
                #     # # plot the activation maps
                #     # plt.imshow(heatmap)
                #     # plt.axis('off')
                #     # plt.title(f'channel {idx+1} heatmap')
                #     # plt.tight_layout()
                #     # plt.show()
                #
                #     # overlaying heatmap and resized input image.
                #     result = (heatmap * 0.5 + img_255 * 0.5) / 255
                #     result_list.append(result)
                #
                #     # plot the overlaying
                #     plt.imshow(result)
                #     plt.axis('off')
                #     plt.title(f'channel {idx + 1} overlay')
                #     plt.tight_layout()
                #     plt.show()

                # channel-wise average the activations
                heatmap = torch.mean(v, 1).squeeze(0).numpy()
                print(f'Max {np.max(heatmap)}   Min {np.min(heatmap)}')
                mask_mask = cv2.resize(heatmap, (i_w, i_h)) < 0
                # heatmap from layers with (H, W) 0-1  # TODO: sigmoid can also do the job!
                # heatmap = 1/(1 + np.exp(-2*heatmap))
                heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # (H, W) 0-1
                heatmap = 1 - heatmap
                # heatmap = np.power(heatmap, 2)
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.resize(heatmap, (i_w, i_h))

                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                b, g, r = cv2.split(heatmap)
                heatmap = cv2.merge([r, g, b])
                result_list.append(heatmap)
                print(heatmap.shape)

                # # plot the activation maps
                # plt.imshow(heatmap)
                # plt.axis('off')
                # plt.tight_layout()
                # plt.show()

                # overlaying heatmap and resized input image.
                result = (heatmap * 0.5 + img_255 * 0.5) / 255
                # result_list.append(result)

                # plot the overlaying
                plt.imshow(result)
                plt.axis('off')
                plt.tight_layout()
                plt.show()  # overlay feature maps on input image

                # # plot the overlaying
                # heatmap_masked = np.where(mask_mask == True, heatmap, 0)
                # plt.imshow(heatmap_masked / 255, cmap="gray")
                # plt.axis('off')
                # plt.tight_layout()
                # plt.show()  # overlay feature maps on input image
                #
                # # Plot the detection results
                # fig = plt.figure()
                # ax2 = fig.add_subplot()
                # plt.imshow(heatmap, cmap='gray')

                # instance segmentation
                if 'masks' in output:
                    proba_threshold = 0.5
                    bool_mask = output['masks'] > proba_threshold
                    bool_mask = bool_mask.squeeze(1)
                    img_mask = draw_segmentation_masks(img_255_tensor, bool_mask, alpha=0.9)
                    plt.imshow(np.asarray(F.to_pil_image(img_mask.detach())))

                plt.axis('off')
                # add boxes
                for idx, (box, score) in enumerate(zip(det_bboxes, det_scores)):
                    rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='b',
                                             facecolor='none')
                    # ax.add_patch(rect)
                    ax2.set_title('Input image with Predictions')
                    ax2.add_patch(rect).set_path_effects([patheffects.Stroke(linewidth=0.5, foreground='black'),
                                                          patheffects.Normal()])
                    ax2.text(box[0], (box[1] - 50), str(round(100 * score.item(), 1)), verticalalignment='top',
                             color='white', fontsize=6.5, weight='bold').set_path_effects([patheffects.Stroke(
                        linewidth=3, foreground='black'), patheffects.Normal()])

                plt.tight_layout()
                plt.show()

                # plot the overlaying
                plt.imshow(np.invert(mask_mask), cmap="gray")
                plt.axis('off')
                plt.tight_layout()
                plt.show()  # overlay feature maps on input image

                # plot the heatmap mask
                heatmap_mask = torch.mean(v, 1).squeeze(0).numpy()
                heatmap_mask = cv2.resize(heatmap_mask, (i_w, i_h))
                heatmap_mask = heatmap_mask > 0

                copy = np.copy(img_255)  # np.copy(img_255)
                copy[heatmap_mask == 0] = 255

                overlay = (0.75 * copy + 0.25 * img_255) / 255

                # plot the overlaying
                plt.imshow(overlay)
                plt.axis('off')
                plt.tight_layout()
                plt.show()

            # Plot the detection results
            fig = plt.figure()
            ax2 = fig.add_subplot()
            plt.imshow(img_255)

            # instance segmentation
            if 'masks' in output:
                proba_threshold = 0.5
                bool_mask = output['masks'] > proba_threshold
                bool_mask = bool_mask.squeeze(1)
                img_mask = draw_segmentation_masks(img_255_tensor, bool_mask, alpha=0.9)
                plt.imshow(np.asarray(F.to_pil_image(img_mask.detach())))

            plt.axis('off')
            # add boxes
            for idx, (box, score) in enumerate(zip(det_bboxes, det_scores)):
                rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='b',
                                         facecolor='none')
                # ax.add_patch(rect)
                ax2.set_title('Input image with Predictions')
                ax2.add_patch(rect).set_path_effects([patheffects.Stroke(linewidth=0.5, foreground='black'),
                                                      patheffects.Normal()])
                ax2.text(box[0], (box[1] - 50), str(round(100 * score.item(), 1)), verticalalignment='top',
                         color='white', fontsize=6.5, weight='bold').set_path_effects([patheffects.Stroke(
                    linewidth=3, foreground='black'), patheffects.Normal()])

            plt.tight_layout()
            plt.show()


def main():
    # get config file
    cfg_path = '.config/eval_config.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)  # conf_thr,

    # register hooks on the rpn module
    rpn_boxes = []
    # model.rpn.register_forward_hook(lambda module, input, output: rpn_boxes.append(output[0]))

    # register hooks on the feature maps
    feature_maps = []
    model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))

    # draw
    transform = GeneralizedRCNNTransform(
        min_size=800,
        max_size=1333,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225]
    )

    transform = GeneralizedRCNNTransform(
        300, 300, image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225], size_divisible=1, fixed_size=(300, 300)
    )

    activation_plot(model, device, val_loader, transform, feature_maps)


if __name__ == '__main__':
    main()
