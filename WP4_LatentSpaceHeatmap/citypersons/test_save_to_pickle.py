import argparse
from collections import OrderedDict
import os
import tqdm
from random import random
import pandas as pd
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
import pickle

from typing import Optional, List, Dict, Tuple, Union
from models import load_model
from data import load_dataloaders
from torchmetrics.classification import BinaryJaccardIndex

def eval_heatmap_metrics(img, paths, gt_boxes, single_heatmap, conf_thresholds=[], intersection_thresholds=[]):



    gt_mask = torch.zeros((len(gt_boxes), img.shape[-2], img.shape[-1])).long()

    for idx, bbox in enumerate(gt_boxes):  # per instance
        x1, y1, x2, y2 = tuple(bbox.to(torch.long))
        gt_mask[idx][y1: y2, x1: x2] = 1




    list_of_res_dicts = []



    res_dict = {}
    res_dict["path"] = paths
    res_dict["annotated_instance"] = {}

    # convert to binary mask w.r.t. the pre-defined conf threshold

    for c in conf_thresholds:
        heatmap_mask = torch.nn.Sigmoid()(single_heatmap) > c
        heatmap_mask = heatmap_mask.long()  # torch tensor with shape (H, W), long type, cpu

        # assert gt_mask.size() == heatmap_mask.size()

        for i in range(gt_mask.shape[0]):

            if i not in res_dict["annotated_instance"]:
                res_dict["annotated_instance"][i] = {}

            # res_dict["annotated_instance"][i]["conf"] = c

            pos_intersection = (gt_mask[i] & heatmap_mask).float().sum()  # TP
            pos_union = (gt_mask[i] | heatmap_mask).float().sum()  # TP+FP+FN

            # res_dict["annotated_instance"][i]["interarea"] = intersection_threshold

            neg_intersection = ((1 - gt_mask[i]) & (1 - heatmap_mask)).float().sum()  # TF
            neg_union = ((1 - gt_mask[i]) | (1 - heatmap_mask)).float().sum()  # TF+FP+FN

            pixels_pos = torch.count_nonzero(gt_mask[i])

            pos_iou = pos_intersection / pos_union
            neg_iou = neg_intersection / neg_union

            if "Pedestrian_IoU_conf" + str(c) not in res_dict["annotated_instance"][i]:
                res_dict["annotated_instance"][i]["Pedestrian_IoU_conf_" + str(c)] = pos_iou.item()

                res_dict["annotated_instance"][i]["Background_IoU_conf_" + str(c)] = neg_iou.item()

            for intersection_threshold in intersection_thresholds:

                interesection_area = pos_intersection / torch.count_nonzero(gt_mask[i])

                res_dict["annotated_instance"][i][
                    "TP_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = False

                res_dict["annotated_instance"][i][
                    "FP_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = False

                res_dict["annotated_instance"][i][
                    "FN_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = False

                if interesection_area > intersection_threshold:
                    tp = True
                    res_dict["annotated_instance"][i][
                        "TP_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = True
                elif interesection_area < intersection_threshold and interesection_area > 0:
                    fp = True
                    res_dict["annotated_instance"][i][
                        "FP_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = True

                elif interesection_area == 0:
                    fn = True
                    res_dict["annotated_instance"][i][
                        "FN_conf_" + str(c) + "_intarea_" + str(intersection_threshold)] = True

    # print(res_dict["annotated_instance"])
    list_of_res_dicts.append(res_dict)

    return list_of_res_dicts
def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def get_detection_TP_FP_FN(detections, labels, conf=0.25, iou_thres=0.45):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        None, updates confusion matrix accordingly
    """

    detections = detections[detections[:, 4] > conf]
    gt_classes = [1]
    #detection_classes = detections[:, 5].int()
    iou = box_iou(labels, detections[:, :4])

    x = torch.where(iou > iou_thres)
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    else:
        matches = np.zeros((0, 3))

    n = matches.shape[0] > 0
    m0, m1, _ = matches.transpose().astype(np.int16)
    gt_res = []
    for i, gc in enumerate(gt_classes):
        j = m0 == i
        gt_res.append({})
        gt_res[i]["TP"] = False
        gt_res[i]["FP"] = False
        gt_res[i]["FN"] = True
        if n and sum(j) == 1:
            # self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
            gt_res[i]["TP"] = True
            gt_res[i]["FN"] = False

        elif not any(m0 == i):
            gt_res[i]["FN"] = True
            gt_res[i]["TP"] = False
            gt_res[i]["FP"] = False
        else:
            # self.matrix[self.nc, gc] += 1  # background FP
            gt_res[i]["FP"] = True
            gt_res[i]["FN"] = False

    # return gt_res

    # if n:
    #     for i, dc in enumerate(detection_classes):
    #         if not any(m1 == i):
    #             gt_res[i]["FN"] = True
    #             gt_res[i]["TP"] = False
    #             gt_res[i]["FP"] = False
    #             #self.matrix[dc, self.nc] += 1  # background FN

    return gt_res

@torch.inference_mode()
def eval_miou_thresholds(model, device, data_loader, feature_maps, conf=0.5, safety_only=False, saving_path='', tag=''):
    model.eval()
    cpu_device = torch.device("cpu")

    # pos_gt_list = []
    # pos_pred_list = []

    pos_intersection_all = 0  # class pedestrian
    neg_intersection_all = 0  # class background

    pos_union_all = 0
    neg_union_all = 0

    n_gt = 0
    n_pred = 0
    df = pd.DataFrame(columns=[
        'conf',
        'mIoU',
        'ped_IoU',
        'bg_IoU',
        'recall',
        'precision',
        'pos_intersection_all',
        'neg_intersection_all',
        'pos_union_all',
        'neg_union_all'
    ])
    final_l = []
    for b, (images, targets) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        if b > 500:
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

        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):

            if len(target['boxes']) == 0:
                continue
            i_h, i_w = img.size()[-2], img.size()[-1]  # size of input image
            # ------------------- safety-relevant instance -------------------
            if safety_only:
                safety = target['ignore']
                gt_boxes = target['boxes'][safety == 0]
            else:
                gt_boxes = target['boxes'].clone()
            # ------------------- creat binary mask ground truth -------------------
            mask = torch.zeros_like(img.cpu()).long()  # (C, H, W)
            for gt in gt_boxes:
                x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
                mask[:, y1: y2, x1: x2] = 1
            gt_mask = mask[0]  # (H, W)
            n_gt += gt_mask.float().sum()

            # pos_gt_list.append(gt_mask)

            # ------------------- creat binary mask prediction -------------------
            activations = feature_maps.pop()
            if isinstance(activations, torch.Tensor):
                activations = OrderedDict([("0", activations)])
            activations = {k: v.detach().cpu() for k, v in activations.items()}

            heatmap_avg = torch.mean(activations['0'], 1)  # (1, H, W)

            resized_heatmap = torchvision.transforms.Resize(size=(i_h, i_w), antialias=True)(heatmap_avg)[0]
            list_of_results = eval_heatmap_metrics(img, '', gt_boxes, resized_heatmap,
                                                   conf_thresholds=[0.4, 0.5, 0.6],
                                                   intersection_thresholds=[0.25, 0.5, 0.75])


            labels = target
            si=i
            nl = len(labels)

            #path = Path(paths[si])
            #seen += 1
            # Predictions
            predn = np.hstack((output['boxes'].cpu().numpy(), np.expand_dims(output['scores'].cpu().numpy(), axis=1), np.expand_dims(output['labels'].cpu().numpy(), axis=1)))
            predn=torch.from_numpy(predn).to(device)
            if nl:
                ### add the dectections per instance here
                res_head = get_detection_TP_FP_FN(predn, target['boxes'], conf=0.25,
                                                  iou_thres=0.45)

                for index in range(len(res_head)):
                    list_of_results[si]["annotated_instance"][index]["TP"] = res_head[index]["TP"]
                    list_of_results[si]["annotated_instance"][index]["FP"] = res_head[index]["FP"]
                    list_of_results[si]["annotated_instance"][index]["FN"] = res_head[index]["FN"]


            final_l.extend(list_of_results)


    ##save pickle object here
    with open(f'{saving_path}/{tag}_stats.p', 'wb') as f:
        pickle.dump(final_l, f)
    print('results saved in:')
    print(f'{saving_path}/{tag}_stats.p')



@torch.inference_mode()
def eval_miou(model, device, data_loader, feature_maps, conf=0.5, safety_only=False):
    model.eval()
    cpu_device = torch.device("cpu")

    # pos_gt_list = []
    # pos_pred_list = []

    pos_intersection_all = 0  # class pedestrian
    neg_intersection_all = 0  # class background

    pos_union_all = 0
    neg_union_all = 0

    n_gt = 0
    n_pred = 0

    for b, (images, targets) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        # if b > 5:
        #     break

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

        for i, (img, target, output) in enumerate(zip(images, targets, outputs)):
            i_h, i_w = img.size()[-2], img.size()[-1]  # size of input image
            # ------------------- safety-relevant instance -------------------
            if safety_only:
                safety = target['ignore']
                gt_boxes = target['boxes'][safety == 0]
            else:
                gt_boxes = target['boxes'].clone()
            # ------------------- creat binary mask ground truth -------------------
            mask = torch.zeros_like(img.cpu()).long()  # (C, H, W)
            for gt in gt_boxes:
                x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
                mask[:, y1: y2, x1: x2] = 1
            gt_mask = mask[0]  # (H, W)
            n_gt += gt_mask.float().sum()

            # pos_gt_list.append(gt_mask)

            # ------------------- creat binary mask prediction -------------------
            activations = feature_maps.pop()
            if isinstance(activations, torch.Tensor):
                activations = OrderedDict([("0", activations)])
            activations = {k: v.detach().cpu() for k, v in activations.items()}

            heatmap_avg = torch.mean(activations['0'], 1)  # (1, H, W)

            resized_heatmap = torchvision.transforms.Resize(size=(i_h, i_w), antialias=True)(heatmap_avg)[0]

            heatmap_mask = torch.nn.Sigmoid()(resized_heatmap) > conf
            heatmap_mask = heatmap_mask.long()  # (H, W)

            # if b <= 4:
            #     plt.imshow(gt_mask, cmap='gray')
            #     plt.title('Gt binary mask')
            #     plt.tight_layout()
            #     plt.axis("off")
            #     plt.show()
            #
            #     plt.imshow(heatmap_mask, cmap='gray')
            #     plt.title('Pred. binary mask')
            #     plt.tight_layout()
            #     plt.axis("off")
            #     plt.show()

            n_pred += heatmap_mask.float().sum()

            # pos_pred_list.append(heatmap_mask)

            # ------------------- IoU -------------------
            pos_intersection = (gt_mask & heatmap_mask).float().sum()  # TP
            pos_union = (gt_mask | heatmap_mask).float().sum()  # TP+FP+FN

            neg_intersection = ((1 - gt_mask) & (1 - heatmap_mask)).float().sum()  # TF
            neg_union = ((1 - gt_mask) | (1 - heatmap_mask)).float().sum()  # TF+FP+FN

            pos_intersection_all += pos_intersection
            neg_intersection_all += neg_intersection

            pos_union_all += pos_union
            neg_union_all += neg_union

    # target = torch.stack(pos_gt_list)
    # preds = torch.stack(pos_pred_list)
    # metric = BinaryJaccardIndex()

    pos_iou = pos_intersection_all / pos_union_all  # metric(preds, target)
    neg_iou = neg_intersection_all / neg_union_all

    miou = (pos_iou + neg_iou) / 2  # we have only two classes: pedestrian and background

    recall = pos_intersection_all / n_gt  # recall of highlighting pedestrians

    precision = pos_intersection_all / n_pred

    print(f'\nSafety-relevant instances only: {safety_only}'
          f'\nUnder {conf} confidence threshold:'
          f'\npedestrian iou: {pos_iou.item():.3f}\nbackground iou: {neg_iou.item():.3f}\nmiou: {miou.item():.3f}'
          f'\nrecall: {recall.item():.3f}'
          f'\nprecision: {precision.item():.3f}')


def main():
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='.config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    parser.add_argument('--conf', default=0.5, help='confidence threshold to get the binary mask ')
    parser.add_argument('--subset', default='test', help='train, val or test')
    parser.add_argument('--safety-only', action='store_true',
                        help='only keep the safety-relevant instances')

    args = parser.parse_args()

    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    root_path = '/'.join(cfg['pretrained_weights_path'].split('/')[:-1])
    contra_weighting_factor = int(root_path.split('/')[-1].split('_')[-1])  # in percentage

    model_name = cfg['model_name'].split('-')[0]  # e.g. FCOS
    dataset_name = cfg['dataset']  # e.g. CityPersons

    tag = f'{dataset_name}_{model_name}_wf_{contra_weighting_factor}'  # e.g. CityPersons_FCOS_wf_10
    print(f'{dataset_name} dataset - {model_name} detector - loss weighting factor {contra_weighting_factor / 100:.3f}')
    pth_name = cfg['pretrained_weights_path'].split('/')[-1].split('.')[0]  # e.g. final_model_coco
    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)
    saving_path=f'{root_path}/{tag}_eval_{pth_name}'
    if args.subset == 'train':
        dataset = train_loader
    elif args.subset == 'val':
        dataset = val_loader
    elif args.subset == 'test':
        dataset = test_loader
    else:
        print(f'Please make sure the subset argument can only take "train", "val" or "test" as input.')
        exit()

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # register hooks on the feature maps
    feature_maps = []
    model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))

    #eval_miou(model, device, dataset, feature_maps, conf=float(args.conf), safety_only=args.safety_only)
    eval_miou_thresholds(model, device, dataset, feature_maps, conf=float(args.conf), safety_only=args.safety_only, saving_path=saving_path, tag=tag)


if __name__ == '__main__':
    main()
