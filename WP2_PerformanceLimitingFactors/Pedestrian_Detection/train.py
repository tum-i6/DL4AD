import argparse
from datetime import datetime
import math
import os
import shutil
import sys
import time

import numpy as np
import torch
import torchvision
import yaml

import data
import models
import utils


models = {
    'KeypointRCNN-ResNet50-KD':
        torchvision.models.detection.keypointrcnn_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2,
            num_keypoints=19
        ),
    'MaskRCNN-ResNet50-IS':
        torchvision.models.detection.maskrcnn_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2
        ),
    'FasterRCNN-ResNet50-OD':
        torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2
        ),
    'FCOS-ResNet50-OD':
        torchvision.models.detection.fcos_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2
        ),
    'RetinaNet-ResNet50-OD':
        torchvision.models.detection.retinanet_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2
        ),
    'SSD300-ResNet50-OD':
        models.ssd300_resnet50(
            pretrained_backbone=True,
            num_classes=2
        )
}


def train_model(cfg, model, device, optimizer, lr_scheduler, train_loader, val_loader, session_path, starting_epoch):
    """
    Optimizes the model weights for several epochs using the dataset from the train_loader. The model's prediction
    performance is evaluated after each epoch, using the validation dataset. Finally, a checkpoint file is stored after
    each epoch containing the model weights, optimizer state, learning rate scheduler state, the model's prediction
    performance on the validation set, the current epoch value and the average prediction losses during that epoch.

    :param cfg: A configuration dictionary containing the amount of epochs for training.
    :param model: The detection model, which should be optimized.
    :param device: The device used during training.
    :param optimizer: An optimizer instance, used for optimizing the model weights.
    :param lr_scheduler: A learning rate scheduler, used for adjusting the learning rate.
    :param train_loader: A dataloader, which holds the train dataset.
    :param val_loader: A dataloader, which holds the validation dataset.
    :param session_path: The training session path, where each checkpoint file should be stored.
    :param starting_epoch: An integer number of the starting epoch.
    """
    if starting_epoch == 0:
        print('--> Start Training <--')
    else:
        print(f'--> Resuming Training at Epoch {starting_epoch} <--')

    for epoch in range(starting_epoch, cfg['num_epochs']):
        print('\n')

        # Trains the model for one epoch
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

        # Evaluates the current model weights on the validation dataset
        eval_results = evaluate_model(model, val_loader, device)

        # Stores the current model weights, optimizer state, learning rate scheduler state, validation set prediction
        # performance, current epoch value and the average prediction losses
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'val_performance': eval_results,
            'epoch': epoch,
            'losses': {}
        }
        for meter in train_metrics.meters:
            if meter == 'lr':
                continue  # Skip the learning rate metric
            checkpoint['losses'][meter] = train_metrics.meters[meter].global_avg
        torch.save(checkpoint, f'{session_path}/checkpoint_epoch_{epoch}.pth')


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    Iterates over all sample images from the data_loader (one epoch) and uses the optimizer to optimize the model
    weights accordingly. Logs the model's train performance every print_freq amount of batches to the console.

    :param model: The detection model, which should be optimized.
    :param optimizer: An optimizer instance, used for optimizing the model weights.
    :param data_loader: A dataloader, which holds the train dataset.
    :param device: The device used during training.
    :param epoch: An integer number of the current epoch.
    :param print_freq: An integer that specifies the amount of processed training batches after which the training logs
    should be printed to the console every time.
    :return: The metric_logger instance that is being used for tracking different sorts of train metrics including the
    prediction losses.
    """
    model.train()  # Sets the model to train mode
    metric_logger = utils.MetricLogger(delimiter="  ")  # Used for tracking different sorts of metrics during training
    # Adds the learning rate as a metric to be tracked
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    lr_scheduler = None  # Used only for the first epoch of training
    # Defines a warmup learning rate scheduler, that is initiated only for the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    # Iterates over all batches from the data_loader
    for images, targets in metric_logger.log_every(data_loader, print_freq, header=f"Epoch: [{epoch}]"):
        # Moves the image samples and target tensors to the GPU device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)  # Infers the images into the model and computes the prediction losses
        losses = sum(loss for loss in loss_dict.values())  # Sums up the prediction losses

        # Checks whether the loss is infinite to stop the training
        if not math.isfinite(losses.item()):
            print(f"Loss is {losses.item()}, stopping training")
            print(loss_dict)
            sys.exit(1)

        # Backpropagates the prediction loss and optimizes the model weights
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            # Applies the learning rate warmup step if the model is in its first epoch
            lr_scheduler.step()

        # Updates the information about the current prediction losses and learning rate value
        metric_logger.update(loss=losses, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def evaluate_train_session(cfg, model, device, test_loader, session_path):
    """
    Finds the checkpoint file (inside the session_path) with the best performing model weights on the validation
    dataset. Loads the checkpoint file and evaluates the model's prediction performance on the test dataset. Stores the
    model weights and the test set prediction performance inside a new checkpoint file called final_model.pth.

    :param cfg: A configuration dictionary containing the amount of epochs for training.
    :param model: The detection model, which should be evaluated.
    :param device: The device used during training.
    :param test_loader: A dataloader, which holds the test dataset.
    :param session_path: The training session path, where each checkpoint file is stored.
    """
    print('\n\n--> Start Final Evaluation <--\n')

    # Determines the checkpoint file, which has achieved the best prediction performance on the validation dataset
    best_performance = 0
    losses = {}  # Used for storing the loss values from all epochs
    val_performance = {}  # Used for storing the val_performance dictionaries from all epochs
    for epoch in range(cfg['num_epochs']):
        checkpoint = torch.load(f'{session_path}/checkpoint_epoch_{epoch}.pth')
        for conf_thr in checkpoint['val_performance']['eval_results']:
            # Checks the F1-score for a specific confidence threshold and an IoU threshold of 0.25 (main metric)
            if checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1'] > best_performance:
                best_performance = checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1']
                best_conf_thr = conf_thr
                best_checkpoint = f'checkpoint_epoch_{epoch}.pth'
                best_epoch = epoch
        # Stores the computed prediction performance on the validation dataset for this epoch
        val_performance[epoch] = checkpoint['val_performance']
        # Stores the computed loss values for this epoch
        for loss in checkpoint['losses']:
            if loss not in losses:
                # Adds the name of the loss metric as a new key
                losses[loss] = [checkpoint['losses'][loss]]
            else:
                # Adds the next value of the loss metric to the list
                losses[loss].append(checkpoint['losses'][loss])

    # Loads the model weights from the best performing checkpoint file
    checkpoint = torch.load(f'{session_path}/{best_checkpoint}')
    model.load_state_dict(checkpoint['model'])

    # Evaluates the model weights from the checkpoint file on the test set
    eval_results = evaluate_model(model, test_loader, device)

    # Stores the final model weights and test set prediction performance inside a dictionary
    checkpoint = {
        'model': model.state_dict(),
        'val_performance': val_performance,
        'test_performance': eval_results,
        'train_losses': losses,
        'final_epoch': best_epoch,
        'conf_thr': best_conf_thr
    }
    torch.save(checkpoint, f'{session_path}/final_model.pth')

    # Prints the final evaluation results with respect to PDSM to the console
    utils.print_test_evaluation_report(f'{session_path}/final_model.pth')


@torch.inference_mode()
def evaluate_model(model, data_loader, device):
    """
    Iterates over all sample images from the data_loader and infers them into the model to evaluate the model's
    prediction performance. Logs the model's inference time and the final evaluation results to the console.

    :param model: The detection model, which should be evaluated.
    :param data_loader: A dataloader, which holds the validation or test dataset.
    :param device: The device used during the evaluation.
    :return: A dictionary that contains the following key-value pairs: 'coco-ap' - contains the average precision value
    computed as specified by the COCO protocol, 'pascal_voc_ap' - contains the average precision value computed as
    specified by the PASCAL VOC protocol, 'lamr' - contains the log average miss rate value as specified by the
    Caltech protocol, 'custom_ap' - contains the average precision value for an IoU threshold of 0.25,
    'custom_lamr' - contains the log average miss rate value for an IoU threshold of 0.25, 'model_time' - contains the
    average model inference time during the evaluation in seconds, 'evaluator_time' - contains the average time for
    evaluating the model predictions in seconds, 'eval_results' - contains a dictionary that contains various confidence
    thresholds as keys. The value of each confidence threshold key is a dictionary that contains various IoU thresholds
    as keys. The value of each Iou threshold key is another dictionary with the amount of TPs (true positives), SRTPs
    (safety relevant true positives), FPs (false positives), FNs (false negatives), the associated precision, recall,
    F1, MR (miss rate) and FPPI (false positives per image) values.
    """
    # Sets the number of CPU threads to 1 during evaluation (this will be reverted at the end of the evaluation)
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    model.eval()  # Sets the model to evaluation mode
    metric_logger = utils.MetricLogger(delimiter="  ")  # Used for tracking different sorts of metrics during evaluation
    # Used for storing the amount of true positives, false positives and false negatives for various IoU and confidence
    # score thresholds
    eval_results = {
        str(conf / 100): {
            str(iou / 100): {
                # Counts the amount of true positives, safety relevant true positives, false positives and
                # false negatives
                'TPs': 0, 'SRTPs': 0, 'FPs': 0, 'FNs': 0
            } for iou in range(25, 101, 5)  # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
        } for conf in range(0, 101, 5)  # Confidence score thresholds from 0.0 to 1.0 with a step size of 0.05
    }

    # Iterates over all batches from the data_loader
    for images, targets in metric_logger.log_every(data_loader, 100, "Test:"):
        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        # Synchronizes the GPU to measure the inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)  # Infers the batch into the model

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time  # Tracks the model's inference time

        evaluator_time = time.time()
        batch_eval_results = evaluate_one_batch(outputs, targets)  # Evaluates the batch of predictions
        # Updates the overall amount of TPs, SRTPs, FPs and FNs
        for conf_threshold in eval_results:
            for iou_threshold in eval_results[conf_threshold]:
                eval_results[conf_threshold][iou_threshold]['TPs'] += \
                    batch_eval_results[conf_threshold][iou_threshold]['TPs']
                eval_results[conf_threshold][iou_threshold]['SRTPs'] += \
                    batch_eval_results[conf_threshold][iou_threshold]['SRTPs']
                eval_results[conf_threshold][iou_threshold]['FPs'] += \
                    batch_eval_results[conf_threshold][iou_threshold]['FPs']
                eval_results[conf_threshold][iou_threshold]['FNs'] += \
                    batch_eval_results[conf_threshold][iou_threshold]['FNs']
        evaluator_time = time.time() - evaluator_time  # Tracks the time for computing the evaluation
        # Updates the information about the model's inference time and the time for computing the evaluation
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # Logs the final stats about the model's inference time and the time for computing the evaluation to the console
    print("Averaged stats:", metric_logger)

    # Iterates over all evaluation thresholds to compute the respective precision, recall, F1-score, MR and FPPI
    for conf_threshold in eval_results:
        for iou_threshold in eval_results[conf_threshold]:
            tps = eval_results[conf_threshold][iou_threshold]['TPs']
            srtps = eval_results[conf_threshold][iou_threshold]['SRTPs']
            fps = eval_results[conf_threshold][iou_threshold]['FPs']
            fns = eval_results[conf_threshold][iou_threshold]['FNs']

            try:
                precision = tps / (tps + fps)
            except ZeroDivisionError:
                precision = 1.0  # This should only occur when conf_threshold = 1.0 (no predictions)

            recall = srtps / (srtps + fns)
            miss_rate = fns / (srtps + fns)
            fppi = fps / len(data_loader.dataset)

            try:
                f1 = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1 = None

            # Stores the computed precision, recall, miss rate, F1-score and fppi values
            eval_results[conf_threshold][iou_threshold]['precision'] = precision
            eval_results[conf_threshold][iou_threshold]['recall'] = recall
            eval_results[conf_threshold][iou_threshold]['F1'] = f1
            eval_results[conf_threshold][iou_threshold]['MR'] = miss_rate
            eval_results[conf_threshold][iou_threshold]['FPPI'] = fppi

    # Computes the average precision values for multiple IoU thresholds (25 to 95 with a step-size of 5)
    aps = utils.get_average_precisions(eval_results)
    coco_ap = round(sum([aps[iou_thr] for iou_thr in aps if float(iou_thr) >= 0.5]) / 10, 3)  # COCO AP metric
    pascal_voc_ap = round(aps['0.5'], 3)  # PASCAL VOC AP metric
    lamr = round(utils.get_log_avg_miss_rate(eval_results, '0.5'), 3)  # LAMR metric
    custom_ap = round(aps['0.25'], 3)  # AP with 0.25 IoU threshold
    custom_lamr = round(utils.get_log_avg_miss_rate(eval_results, '0.25'), 3)  # LAMR with 0.25 IoU threshold
    # Stores the computed metric values
    performance_summary = {
        'coco_ap': coco_ap,
        'pascal_voc_ap': pascal_voc_ap,
        'lamr': lamr,
        'custom_ap': custom_ap,
        'custom_lamr': custom_lamr,
        'model_time': metric_logger.meters['model_time'].global_avg,
        'evaluator_time': metric_logger.meters['evaluator_time'].global_avg,
        'eval_results': eval_results,
    }

    print('\nPrediction Performance Summary')
    print('Standard Metrics:')
    print(f'Average Precision           (AP)    @[ IoU=0.5:0.05:0.95  &  C=0.0:0.05:1.0 ] = {coco_ap}')
    print(f'Average Precision           (AP)    @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {pascal_voc_ap}')
    print(f'Log-Average Miss Rate       (LAMR)  @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {lamr}')
    print('\nCustomized Task Specific Metrics:')
    print(f'Average Precision           (AP)    @[ IoU=0.25           &  C=0.0:0.05:1.0 ] = {custom_ap}')
    print(f'Log-Average Miss Rate       (LAMR)  @[ IoU=0.25           &  C=0.0:0.05:1.0 ] = {custom_lamr}')
    print(f'Miss Rate                   (MR)    @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["MR"], 3)}')
    print(f'False Positives per Image   (FPPI)  @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["FPPI"], 3)}')
    print(f'F1 Score                    (F1)    @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["F1"], 3)}')
    print(f'Precision                   (P)     @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["precision"], 3)}')
    print(f'Recall                      (R)     @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["recall"], 3)}')

    torch.set_num_threads(n_threads)  # Sets the number of CPU threads as it was before the evaluation
    return performance_summary


def evaluate_one_batch(outputs, targets):
    """
    Evaluates the detection performance for a batch of image samples over multiple IoU and confidence score thresholds.
    Note that the class labels are not being evaluated, since it is being assumed that only one class exists within
    the outputs and the targets (the pedestrian/human class). Note that ground truth annotations with the iscrowd
    property set 1 can be matched multiple times, since they represent a group of people and not a single pedestrian
    instance.

    :param outputs: A PyTorch tensor that contains the model predictions for the batch.
    :param targets: A PyTorch tensor that contains the ground truth labels for the batch.
    :return: A dictionary that contains various confidence thresholds as keys. The value of each confidence threshold
    key is a dictionary that contains various IoU thresholds as keys. The value of each IoU threshold key is another
    dictionary with the amount of TPs (true positives), SRTPs (safety relevant true positives), FPs (false positives)
    and FNs (false negatives).
    """
    # Used for storing the amount of true positives, safety relevant true positives, false positives and false negatives
    # for various IoU and confidence score thresholds on a batch level
    batch_eval_results = {
        str(conf/100): {
            str(iou/100): {
                # Counts the amount of true positives, safety relevant true positives, false positives and
                # false negatives
                'TPs': 0, 'SRTPs': 0, 'FPs': 0, 'FNs': 0
            } for iou in range(25, 101, 5)  # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
        } for conf in range(0, 101, 5)  # Confidence score thresholds from 0.0 to 1.0 with a step size of 0.05
    }

    # Iterates over all batch items
    for batch_idx in range(len(targets)):
        # Extracts the ground truth information for each object
        gts = {}
        amount_safety_relevant_peds = 0  # Tracks the amount of pedestrians with the ignore property set to 0
        for j in range(len(targets[batch_idx]['boxes'])):
            gts[j] = {
                'bbox': targets[batch_idx]['boxes'][j],
                #'label': targets[batch_idx]['labels'][j],
                'area': targets[batch_idx]['area'][j],
                'ignore': targets[batch_idx]['ignore'][j]
            }
            # Extracts the information about the iscrowd property (if it exists)
            if 'iscrowd' in targets[batch_idx]:
                gts[j]['iscrowd'] = targets[batch_idx]['iscrowd'][j]
            if targets[batch_idx]['ignore'][j] == 0:
                amount_safety_relevant_peds += 1

        # Extracts the predictions for each detected object
        dts = {}
        for j in range(len(outputs[batch_idx]['boxes'])):
            dts[j] = {
                'bbox': outputs[batch_idx]['boxes'][j],
                #'label': outputs[batch_idx]['labels'][j],
                'score': outputs[batch_idx]['scores'][j]
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

        # Iterates over all confidence thresholds from the batch_eval_results dictionary
        for conf_thr in batch_eval_results:
            # Creates a copy of the gts_dts_iou matrix in order to filter it w.r. to the conf_thr
            gts_dts_ious_conf_thr = gts_dts_ious.copy()
            dt_amount = 0  # Counts the amount of detections after applying the conf_thr
            for dt_id in dts:
                if dts[dt_id]['score'].item() < float(conf_thr):
                    # Resets the IoU values for detections with a confidence score below the threshold
                    gts_dts_ious_conf_thr[:, dt_id] = 0
                else:
                    dt_amount += 1

            # Used for storing the amount of true positives, safety relevant true positives, false positives and false
            # negatives for various IoU thresholds on a single image sample level (with each TP, the amount of FPs will
            # be decreased, the same applies for the FNs if the TP was a safety relevant pedestrian -> SRTP)
            img_eval_results = {
                str(iou/100): {
                    # Initially, all detections are considered as FPs and all safety relevant pedestrians are
                    # considered as FNs
                    'TPs': 0, 'SRTPs': 0, 'FPs': dt_amount, 'FNs': amount_safety_relevant_peds
                } for iou in range(25, 101, 5)  # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
            }

            matched_gts = []  # A list used for tracking the gt_ids of matched bboxes
            # Matches the detections to the ground truth objects and fills the img_eval_results dictionary
            while True:
                if not np.any(gts_dts_ious_conf_thr):
                    # Exit the loop if the matrix is an empty sequence
                    break

                # Matrix coordinates of a ground truth and prediction bbox with the currently highest IoU
                match = np.unravel_index(gts_dts_ious_conf_thr.argmax(), gts_dts_ious_conf_thr.shape)

                if gts_dts_ious_conf_thr[match[0], match[1]] < 0.25:
                    # Exit the loop if the highest IoU value is lower than the lowest IoU threshold
                    break

                # Computes the amount of TPs, FPs and FNs with respect to the IoU thresholds from img_eval_results
                for iou_threshold in img_eval_results:
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

                matched_gts.append(match[0])  # Adds the gt_id to the list of matched ground truths

            # Updates the batch level amount of TPs, SRTPs, FPs and FNs
            for iou_threshold in batch_eval_results[conf_thr]:
                batch_eval_results[conf_thr][iou_threshold]['TPs'] += img_eval_results[iou_threshold]['TPs']
                batch_eval_results[conf_thr][iou_threshold]['SRTPs'] += img_eval_results[iou_threshold]['SRTPs']
                batch_eval_results[conf_thr][iou_threshold]['FPs'] += img_eval_results[iou_threshold]['FPs']
                batch_eval_results[conf_thr][iou_threshold]['FNs'] += img_eval_results[iou_threshold]['FNs']

    return batch_eval_results


def load_dataloaders(cfg):
    """
    Loads the dataloaders for the selected dataset.

    :param cfg: A configuration dictionary containing the name of the selected dataset.
    :return: A train, validation and test dataloader instance, used for training and evaluating the detection model.
    """
    if cfg['dataset'] == 'KIA':
        train_loader, val_loader, test_loader = data.load_kia_dataloaders(cfg)
    elif cfg['dataset'] == 'CityPersons':
        train_loader, val_loader, test_loader = data.load_citypersons_dataloaders(cfg)
    return train_loader, val_loader, test_loader


def load_optimizer(cfg, model):
    """
    Loads the optimizer and learning rate scheduler for optimizing the weights of the detection model and for
    dynamically adjusting the learning rate.

    :param cfg: A configuration dictionary containing the name of the optimizer, learning rate, momentum, weight decay,
    lr_scheduler_step_size and lr_scheduler_gamma value.
    :param model: The detection model, which should be trained using the optimizer.
    :return: An optimizer instance (SGD or Adam) and a learning rate scheduler instance.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]  # List of trainable model weight parameters
    # Sets the optimizer as Stochastic Gradient Descent
    if(cfg['optimizer']) == 'SGD':
        optimizer = torch.optim.SGD(
            trainable_params,
            cfg['learning_rate'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay']
        )
    # Sets the optimizer as Adam
    elif(cfg['optimizer']) == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            cfg['learning_rate'],
            weight_decay=cfg['weight_decay']
        )

    # Instantiates the learning rate scheduler for decreasing the learning rate every step_size epochs by gamma amount
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['lr_scheduler_step_size'],
        gamma=cfg['lr_scheduler_gamma']
    )

    return optimizer, lr_scheduler


def load_model(cfg):
    """
    Loads the detection model and the device, used for training/evaluation. Note that only one GPU can be utilized at
    the time.

    :param cfg: A configuration dictionary containing the name of the detection model, the GPU ID and optionally a
    string path to a model weights file, which should be used for fine-tuning.
    :return: The detection model and the device, which are being used for training/evaluation.
    """
    # Sets the GPU, which should be used for training/evaluation
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_id'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\nDevice used for training/evaluation: {device}')
    print(f"\nLoading {cfg['model_name']} model.")
    # Loads the selected detection model
    model = models[cfg['model_name']]  # Instantiates the model
    model.to(device)  # Moves the model to the device
    print(cfg['model_name'] + ' has been loaded!\n')
    # Loads the pretrained model weights, if specified
    if 'pretrained_weights_path' in cfg:
        if os.path.exists(cfg['pretrained_weights_path']):
            checkpoint = torch.load(cfg['pretrained_weights_path'])
            model.load_state_dict(checkpoint['model'])
    return model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to a train session folder, which was previously created by this script. Used to '
                             'resume training sessions, that did not finish.')
    args = parser.parse_args()

    if args.resume:
        # Reads the yaml file, from the train session folder, which has been created by a previous training run
        with open(f'{args.resume}/train_config.yaml') as yaml_file:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Determines at which epoch the training should resume
        last_epoch = 0
        train_files = os.listdir(args.resume)
        for file in train_files:
            if file.startswith('checkpoint'):
                epoch_checkpoint = int(file.split('_')[-1][:-4])
                if epoch_checkpoint > last_epoch:
                    last_epoch = epoch_checkpoint
        if last_epoch+1 == cfg['num_epochs']:
            print(f'Training cannot be resumed, since it already finished at epoch {last_epoch} (counting from 0).')
            sys.exit(1)
    else:
        # Reads the yaml file, which contains the train parameters
        with open(args.config) as yaml_file:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # Defines the training session name and creates a new sub-folder for storing the training results and log files
        session_name = \
            f"{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}_{cfg['dataset']}_{cfg['model_name']}"
        session_path = f"{cfg['output_path']}/{session_name}"
        os.mkdir(session_path)

        print(f"Initiating {cfg['dataset']} 2D detection training for the {cfg['model_name']} model.")

        # Displays all parameters for the current training session and creates a copy of the respective yaml file
        shutil.copyfile(args.config, f"{session_path}/train_config.yaml")
        print('\nPARAMETERS')
        for param in cfg.keys():
            print('    ' + param + ' = ' + str(cfg[param]))

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)
    # Loads the detection model and the device
    model, device = load_model(cfg)
    # Loads the optimizer and learning rate scheduler
    optimizer, lr_scheduler = load_optimizer(cfg, model)

    if args.resume:
        # Loads the latest checkpoint file and overwrites the model weights, optimizer and learning rate scheduler
        checkpoint = torch.load(f'{args.resume}/checkpoint_epoch_{last_epoch}.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        starting_epoch = last_epoch + 1  # Sets the new starting epoch
        session_path = args.resume  # Sets the session path from which the training resumes
    else:
        starting_epoch = 0  # Sets the baseline starting epoch

    # Initiates the training of the detection model
    train_model(cfg, model, device, optimizer, lr_scheduler, train_loader, val_loader, session_path, starting_epoch)
    # Initiates the evaluation of the detection model
    evaluate_train_session(cfg, model, device, test_loader, session_path)


if __name__ == '__main__':
    main()
