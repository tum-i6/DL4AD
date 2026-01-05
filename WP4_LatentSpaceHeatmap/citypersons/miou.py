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

from typing import Optional, List, Dict, Tuple, Union
from models import load_model
from data import load_dataloaders
from torchmetrics.classification import BinaryJaccardIndex

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
    for b, (images, targets) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader)):
        #if b > 5000:
        #    break

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
            for conf_indx, conf_thr in enumerate(range(0, 101, 5)):
                conf_thr=conf_thr/100
                heatmap_mask = torch.nn.Sigmoid()(resized_heatmap) > conf_thr
                heatmap_mask = heatmap_mask.long()  # (H, W)
                n_pred += heatmap_mask.float().sum()

                # pos_pred_list.append(heatmap_mask)

                # ------------------- IoU -------------------
                pos_intersection = (gt_mask & heatmap_mask).float().sum()  # TP
                pos_union = (gt_mask | heatmap_mask).float().sum()  # TP+FP+FN

                neg_intersection = ((1 - gt_mask) & (1 - heatmap_mask)).float().sum()  # TF
                neg_union = ((1 - gt_mask) | (1 - heatmap_mask)).float().sum()  # TF+FP+FN

                #pos_intersection_all += pos_intersection
                if len(df)!=21:
                    new_row={
                        'conf':conf_thr,
                        'pos_intersection_all':pos_intersection.item(),
                        'neg_intersection_all': neg_intersection.item(),
                        'pos_union_all': pos_union.item(),
                        'neg_union_all': neg_union.item()
                    }
                    df.loc[conf_indx]=new_row
                else:
                    #df.at[conf_indx, 'conf']=conf_thr
                    df.loc[conf_indx]['pos_intersection_all']+=pos_intersection.item()
                    df.loc[conf_indx]['neg_intersection_all']+=neg_intersection.item()
                    df.loc[conf_indx]['pos_union_all']+=pos_union.item()
                    df.loc[conf_indx]['neg_union_all']+=neg_union.item()
                #neg_intersection_all += neg_intersection


                #pos_union_all += pos_union
                #neg_union_all += neg_union

    # target = torch.stack(pos_gt_list)
    # preds = torch.stack(pos_pred_list)
    # metric = BinaryJaccardIndex()
    for index, row in df.iterrows():
        pos_iou = row['pos_intersection_all']/row['pos_union_all']
        neg_iou = row['neg_intersection_all']/row['neg_union_all']
        miou = (pos_iou + neg_iou) / 2
        recall = row['pos_intersection_all'] / n_gt  # recall of highlighting pedestrians
        precision = row['pos_intersection_all'] / n_pred

        row['mIoU']=miou.item()
        row['ped_IoU']=pos_iou.item()
        row['bg_IoU']=neg_iou.item()
        row['recall']=recall.item()
        row['precision']=precision.item()


        print(f'\nSafety-relevant instances only: {safety_only}'
              f'\nUnder {row["conf"]} confidence threshold:'
              f'\npedestrian iou: {pos_iou.item():.3f}\nbackground iou: {neg_iou.item():.3f}\nmiou: {miou.item():.3f}'
              f'\nrecall: {recall.item():.3f}'
              f'\nprecision: {precision.item():.3f}')
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    df.to_pickle(f'{saving_path}/{tag}_heatmap_quality_test_set.p')
    print(f'The results were saved in: {saving_path}/{tag}_heatmap_quality_test_set.p')


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
