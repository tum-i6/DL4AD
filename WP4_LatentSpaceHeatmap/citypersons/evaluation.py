import argparse
import os
import cv2
import torch
import yaml
from matplotlib import pyplot as plt
import pandas as pd
from roi_plf_extractor import extract_plf
from visul_feat_map import visul_heatmap
from models import load_model, standard_conf_thres
from data import load_dataloaders
import warnings
import time
import numpy as np
import utils
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import csv
from plf_analysis_plot import plot_average_precisions, plot_log_avg_miss_rate, plot_AUC_precision_recall, plot_score_distributions

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
def boost_scores_method4(prob_map, output, alpha, beta, bias=0, conf_threshold=0.05):
    copied_output = output.copy()
    prob_map = np.where(prob_map > .25, prob_map, 0.0)
    # convert boxes to integer type
    output['boxes'] = torch.round(output['boxes']).type(torch.int64)

    adjusted_scores = []

    for origi_box, box, score in zip(copied_output['boxes'], output['boxes'], output['scores']):
        x1, y1, x2, y2 = tuple(box.numpy())

        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            averaged_score = score.item()
        else:
            # averaged_score = np.sum(np.power(prob_map[y1:y2, x1:x2], 2) * 2 * score.item()) / area  # (1
            # averaged_score = np.sum(np.sqrt(np.power(prob_map[y1:y2, x1:x2], 2) * score.item())) / area  # (2
            # averaged_score = np.sum(prob_map[y1:y2, x1:x2] * score.item()) / area  # (3
            # averaged_score = np.mean(prob_map[y1:y2, x1:x2]) / (1 + np.exp(-(score.item() - c) * b))  # (4
            #averaged_score = score.item() / (1 + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - beta) * alpha))  # (5
            averaged_score = (score.item() + 2*bias) / ( 1 + bias + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - beta) * alpha)) # (6 with bias
            #averaged_score = (score.item() + 2*bias) / ( 1  + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - bias) * alpha)) # (7 with bias
            #averaged_score = (score.item() + np.mean(prob_map[y1:y2, x1:x2]) ) / ( 1  + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - bias) * alpha)) # (7 with bias
            averaged_score = max(0, min(averaged_score, 1))

            # avg_prob = np.sum(prob_map[y1:y2, x1:x2]) / area
            # # if score.item() < .5 < avg_prob:  # low-confident predictions in pedestrian area
            # #     # alternatives:
            # #     # np.sum(prob_map[y1:y2, x1:x2] * 2 * score.item()) / area
            # #     # (score.item() + avg_prob) / 2
            # #     averaged_score = np.sum(np.power(prob_map[y1:y2, x1:x2], 2) * 2 * score.item()) / area
            # #     averaged_score = max(0, min(averaged_score, 1))
            # if np.power(avg_prob, 2) < 0.5:  # any predictions outside of the pedestrian areas
            #     # averaged_score = np.sum(np.power(prob_map[y1:y2, x1:x2], 2) * 2 * score.item()) / area
            #     averaged_score = np.sum(np.sqrt(np.power(prob_map[y1:y2, x1:x2], 2) * score.item())) / area
            #     averaged_score = max(0, min(averaged_score, 1))
            # else:
            #     averaged_score = score.item()

        adjusted_scores.append(averaged_score)

    copied_output['scores'] = torch.tensor(adjusted_scores, dtype=output['scores'].dtype)

    copied_output = {k: v[copied_output['scores'] > conf_threshold] for k, v in copied_output.items()}

    return copied_output

def boost_scores(prob_map, output, conf_threshold=0.05):
    """
    the modified post-processing step for the model trained with the new loss to adjust the confidence scores.
    :param prob_map: the normalized probability map with the same size as the input image
    :param output: the original predictions without any confidence threshold
    :param conf_threshold: the default confidence to finally eliminate the low-confidence predictions
    :return: the prediction dict after modifying the confidence scores
    """
    copied_output = output.copy()

    # convert boxes to integer type
    output['boxes'] = torch.round(output['boxes']).type(torch.int64)

    # a placeholder list to store the adjusted scores
    adjusted_scores = []

    # iterate each original prediction
    for origi_box, box, score in zip(copied_output['boxes'], output['boxes'], output['scores']):
        x1, y1, x2, y2 = tuple(box.numpy())
        # the area of the predictions
        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            averaged_score = score.item()  # keep the original score
        else:
            # averaged_score = np.sum(np.power(prob_map[y1:y2, x1:x2], 2) * 2 * score.item()) / area  # (1
            # averaged_score = np.sum(np.sqrt(np.power(prob_map[y1:y2, x1:x2], 2) * score.item())) / area  # (2
            averaged_score = np.sum(prob_map[y1:y2, x1:x2] * score.item()) / area  # (3
            # averaged_score = np.mean(prob_map[y1:y2, x1:x2]) / (1 + np.exp(-(score.item() - 0.2) * 5))  # (4
            # averaged_score = score.item() / (1 + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - 0.2) * 5))  # (5

            # clamp the adjusted confidence into the range 0-1
            averaged_score = max(0, min(averaged_score, 1))

        adjusted_scores.append(averaged_score)
    # update the adjusted scores
    copied_output['scores'] = torch.tensor(adjusted_scores, dtype=output['scores'].dtype)
    # eliminate the low-confidence predictions to keep the fair comparison with the baseline
    copied_output = {k: v[copied_output['scores'] > conf_threshold] for k, v in copied_output.items()}

    return copied_output


@torch.inference_mode()
def evaluate_model(model, data_loader, device, conf_threshold_post_pros=None, feat_maps=None, cfg=None):
    """
    Iterates over all sample images from the data_loader and infers them into the model to evaluate the model's
    prediction performance. Logs the model's inference time and the final evaluation results to the console.

    :param model: The detection model, which should be evaluated.
    :param data_loader: A dataloader, which holds the validation or test dataset.
    :param device: The device used during the evaluation.
    :param conf_threshold_post_pros: the default confidence used by the model. Only for the modified post-processing!
    :param feat_maps: a list storing the extracted feature maps

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
    scores={
        "before_tp":[],
        "before_fp":[],
        "after_tp":[],
        "after_fp":[]
    }
    # Iterates over all batches from the data_loader
    #for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, "Test:")):
    for batch_idx, (images, targets) in tqdm(enumerate(data_loader), total=min(5000,len(data_loader))):
        if batch_idx>5000:
            break

        # Moves the image samples to the GPU
        images = [img.to(device) for img in images]

        # Synchronizes the GPU to measure the inference time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)  # Infers the batch into the model

        # Moves the predictions from the GPU to the CPU
        outputs = [{k: v.to(cpu_device) for k, v in t.items() if k != 'masks'} for t in outputs]

        # update the final detection results by utilizing the probability map
        if feat_maps is not None:
            # get feature maps
            activations = feat_maps.pop()

            # only keep the first FPN layer activation
            if cfg['model_name'] != 'SSD300-ResNet50-OD':
                activations = {k: v.detach().cpu() for k, v in activations.items() if k == '0'}  # shape 1 C H W
            else:
                activations=activations.detach().cpu()

            if len(images) != 1:
                print('Please set the batch number for evaluation to 1')
                exit()
            else:
                img = images[0]
                output = outputs[0]
                # original input image
                original_h, original_w = img.size()[-2:]
                i_h, i_w = tuple([original_h, original_w])

                # channel-wise average & resize to input image space
                if cfg['model_name'] != 'SSD300-ResNet50-OD':
                    heatmap = torch.mean(activations['0'], 1).squeeze(0).numpy()
                else:
                    heatmap = torch.mean(activations, 1).squeeze(0).numpy()
                resized_heatmap = cv2.resize(heatmap, (i_w, i_h))

                # probability/ likelihood map with value 0-1 by sigmoid func
                p_map = 1 / (1 + np.exp(-resized_heatmap * 2))
                eval_results_before_boosting, tp_scores, fp_scores=evaluate_one_batch_main(outputs, targets)
                scores['before_tp'].extend(tp_scores)
                scores['before_fp'].extend(fp_scores)
                # update the scores
                assert conf_threshold_post_pros is not None
                #outputs = [boost_scores(p_map, output, conf_threshold=conf_threshold_post_pros)]
                #scores['before'].extend(output['scores'].cpu().detach().numpy())
                outputs = [boost_scores_method4(p_map, output,cfg['b'],cfg['c'], cfg['bias'], conf_threshold=conf_threshold_post_pros)]
                #scores['after'].extend(outputs[0]['scores'].cpu().detach().numpy())

        targets = [{k: v for k, v in t.items() if k != 'masks'} for t in targets]

        model_time = time.time() - model_time  # Tracks the model's inference time

        evaluator_time = time.time()
        batch_eval_results = evaluate_one_batch(outputs, targets)  # Evaluates the batch of predictions
        eval_results_before_boosting, tp_scores, fp_scores = evaluate_one_batch_main(outputs, targets)
        scores['after_tp'].extend(tp_scores)
        scores['after_fp'].extend(fp_scores)
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
                f1 = None  # no FPs at all: all missed

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
    custom_ap = round(aps['0.5'], 3)  # AP with 0.25 IoU threshold
    custom_lamr = round(utils.get_log_avg_miss_rate(eval_results, '0.5'), 3)  # LAMR with 0.25 IoU threshold
    roc=utils.get_ROC(eval_results, iou_threshold="0.5")
    mean_recall = 0
    mean_precision = 0
    for conf in list(range(0, 101, 5)):
        conf /= 100
        mean_recall += eval_results[str(conf)]["0.5"]["recall"]
        mean_precision += eval_results[str(conf)]["0.5"]["precision"]
    mean_recall /= len(range(0, 101, 5))
    mean_precision /= len(range(0, 101, 5))

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
        'roc':roc
    }

    print('\nPrediction Performance Summary')
    print('Standard Metrics:')
    print(f'Average Precision           (AP)    @[ IoU=0.5:0.05:0.95  &  C=0.0:0.05:1.0 ] = {coco_ap}')
    print(f'Average Precision           (AP)    @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {pascal_voc_ap}')
    print(f'Log-Average Miss Rate       (LAMR)  @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {lamr}')
    print(f'AuRoc Precision vs. Recall  (AuRoc) @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {roc["auc_score"]}')

    print('\nCustomized Task Specific Metrics:')
    print(f'Average Precision           (AP)    @[ IoU=0.5           &  C=0.0:0.05:1.0 ] = {custom_ap}')
    print(f'Log-Average Miss Rate       (LAMR)  @[ IoU=0.5           &  C=0.0:0.05:1.0 ] = {custom_lamr}')
    print(f'Mean Recall                 (R)     @[ IoU=0.5           &  C=0.0:0.05:1.0 ] = {round(mean_recall, 3)}')
    print(
        f'Mean Precision              (P)     @[ IoU=0.5           &  C=0.0:0.05:1.0 ] = {round(mean_precision, 3)}')
    print(
        f'Miss Rate                   (MR)    @[ IoU=0.5           &  C=0.5          ] = {round(eval_results["0.5"]["0.5"]["MR"], 3)}')
    print(
        f'False Positives per Image   (FPPI)  @[ IoU=0.5           &  C=0.5          ] = {round(eval_results["0.5"]["0.5"]["FPPI"], 3)}')
    print(
        f'F1 Score                    (F1)    @[ IoU=0.5           &  C=0.5          ] = {round(eval_results["0.5"]["0.5"]["F1"], 3)}')
    print(
        f'Precision                   (P)     @[ IoU=0.5           &  C=0.5          ] = {round(eval_results["0.5"]["0.5"]["precision"], 3)}')
    print(
        f'Recall                      (R)     @[ IoU=0.5           &  C=0.5          ] = {round(eval_results["0.5"]["0.5"]["recall"], 3)}')

    torch.set_num_threads(n_threads)  # Sets the number of CPU threads as it was before the evaluation
    return performance_summary, scores
def evaluate_one_batch_main(outputs, targets):
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
        str(conf / 100): {
            str(iou / 100): {
                # Counts the amount of true positives, safety relevant true positives, false positives and
                # false negatives
                'TPs': 0, 'SRTPs': 0, 'FPs': 0, 'FNs': 0
            } for iou in range(25, 101, 5)  # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
        } for conf in range(0, 101, 5)  # Confidence score thresholds from 0.0 to 1.0 with a step size of 0.05, 21 in total
    }

    tps_conf_scores = []  # List to store confidence scores for true positives
    fps_conf_scores = []  # List to store confidence scores for false positives

    # Iterates over all batch items
    for batch_idx in range(len(targets)):
        # Extracts the ground truth information for each object
        gts = {}
        amount_safety_relevant_peds = 0  # Tracks the amount of pedestrians with the ignore property set to 0
        for j in range(len(targets[batch_idx]['boxes'])):
            gts[j] = {
                'bbox': targets[batch_idx]['boxes'][j],
                # 'label': targets[batch_idx]['labels'][j],
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
                # 'label': outputs[batch_idx]['labels'][j],
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
                str(iou / 100): {
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
                        tps_conf_scores.append(dts[match[1]]['score'].item())
                    else:
                        fps_conf_scores.append(dts[match[1]]['score'].item())

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

    return batch_eval_results, tps_conf_scores, fps_conf_scores
def heatmaps_main(outputs, targets, heatmaps):
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
        str(conf / 100): {
            # Counts the amount of true positives, safety relevant true positives, false positives and
            # false negatives
            'HTPs': 0 # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
        } for conf in range(0, 101, 5)  # Confidence score thresholds from 0.0 to 1.0 with a step size of 0.05, 21 in total
    }


    # Iterates over all batch items
    for batch_idx in range(len(targets)):
        # Extracts the ground truth information for each object
        gts = {}
        amount_safety_relevant_peds = 0  # Tracks the amount of pedestrians with the ignore property set to 0
        for j in range(len(targets[batch_idx]['boxes'])):
            gts[j] = {
                'bbox': targets[batch_idx]['boxes'][j],
                # 'label': targets[batch_idx]['labels'][j],
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
                # 'label': outputs[batch_idx]['labels'][j],
                'score': outputs[batch_idx]['scores'][j]
            }

        # A matrix that stores the IoU value between each ground truth and predicted bbox
        gts_dts_ious = np.zeros((len(gts), len(dts)))

        # Checks whether there are any ground truths or predictions
        if len(gts) != 0:
            for mask_confs in batch_eval_results:
                # Computes the IoU between each ground truth and each predicted bbox
                binary_mask = np.where(heatmap >= int(mask_confs), 1, 0)
                for gt_id in gts:
                    # Crop the mask based on the bounding box coordinates
                    x1 = gts[gt_id]['bbox'][0]
                    y1 = gts[gt_id]['bbox'][1]
                    x2 = gts[gt_id]['bbox'][2]
                    y2 = gts[gt_id]['bbox'][3]
                    cropped_mask = binary_mask[y1:y2, x1:x2]
                    inter_area = np.count_nonzero(cropped_mask)
                    if inter_area == 0:
                        # No intersection -> IoU is 0
                        gts_dts_ious[gt_id][dt_id] = 0
                        continue


                    # Computes the area of the predicted bbox
                    detection_area = abs((y2-y1)*(x2-x1))
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
                str(iou / 100): {
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
                        tps_conf_scores.append(dts[match[1]]['score'].item())
                    else:
                        fps_conf_scores.append(dts[match[1]]['score'].item())

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

    return batch_eval_results, tps_conf_scores, fps_conf_scores

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
        str(conf / 100): {
            str(iou / 100): {
                # Counts the amount of true positives, safety relevant true positives, false positives and
                # false negatives
                'TPs': 0, 'SRTPs': 0, 'FPs': 0, 'FNs': 0
            } for iou in range(25, 101, 5)  # IoU thresholds from 0.25 to 1.0 with a step size of 0.05
        } for conf in range(0, 101, 5)  # Confidence score thresholds from 0.0 to 1.0 with a step size of 0.05, 21 in total
    }

    # Iterates over all batch items
    for batch_idx in range(len(targets)):
        # Extracts the ground truth information for each object
        gts = {}
        amount_safety_relevant_peds = 0  # Tracks the amount of pedestrians with the ignore property set to 0
        for j in range(len(targets[batch_idx]['boxes'])):
            gts[j] = {
                'bbox': targets[batch_idx]['boxes'][j],
                # 'label': targets[batch_idx]['labels'][j],
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
                # 'label': outputs[batch_idx]['labels'][j],
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
                str(iou / 100): {
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


def main():
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='.config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    parser.add_argument('--subset', default='test', help='train, val or test')

    #parser.add_argument('--mode', default='boosted', help='boosted, normal')

    parser.add_argument('--save-vis', action='store_true',
                        help='save a couple of example images with detections')
    parser.add_argument('--eval', action='store_true', default=True,
                        help='don`t eval the model')
    parser.add_argument('--plfs', action='store_true',
                        help='save the dataframe containing the PLFs')

    args = parser.parse_args()

    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)

    # specify which subset will be used for eval
    if args.subset == 'train':
        dataset = train_loader
    elif args.subset == 'val':
        dataset = val_loader
    elif args.subset == 'test':
        dataset = test_loader
    else:
        print(f'Please make sure the subset argument can only take "train", "val" or "test" as input.')
        exit()

    subset_name = f'{args.subset}_set'

    # infer the root path and the extra infos
    root_path = '/'.join(cfg['pretrained_weights_path'].split('/')[:-1])
    contra_weighting_factor = int(root_path.split('/')[-1].split('_')[-1])  # in percentage

    model_name = cfg['model_name'].split('-')[0]  # e.g. FCOS
    dataset_name = cfg['dataset']  # e.g. CityPersons

    tag = f'{dataset_name}_{model_name}_wf_{contra_weighting_factor}'  # e.g. CityPersons_FCOS_wf_10
    print(f'{dataset_name} dataset - {model_name} detector - loss weighting factor {contra_weighting_factor/100:.3f}')
    print(f'Checkpoint: {cfg["pretrained_weights_path"]}')

    # infer the name of the evaluated model
    pth_name = cfg['pretrained_weights_path'].split('/')[-1].split('.')[0]  # e.g. final_model_coco

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # ------------------------- part 1: evaluation -------------------------

    if args.eval:
        # creat a folder to store the eval curves such as precision vs. recall curves
        if not os.path.exists(f'{root_path}/{tag}_eval_{pth_name}/curves'):
            os.makedirs(f'{root_path}/{tag}_eval_{pth_name}/curves')

        if cfg['mode']=='normal':
            print(f'\nstart standard evaluating...')
            # Evaluates the model weights from the checkpoint file on the test set
            results, _ = evaluate_model(model, dataset, device)
            # save the eval results into a json file
            with open(f'{root_path}/{tag}_eval_{pth_name}/{tag}_eval_result_{subset_name}.json', 'w') as fp:
                json.dump(results, fp)
            print(f'\nstandard eval results have been saved!')

            # start saving the eval curves
            results_dict = {f'{model_name}': results['eval_results']}
            print(f'\nsaving curves...')
            for iou in ['0.25', '0.5', '0.75']:
                fig_mr = plot_log_avg_miss_rate(results_dict, iou, plot=False)
                fig_pr = plot_average_precisions(results_dict, iou, plot=False)

                fig_mr.savefig(f'{root_path}/{tag}_eval_{pth_name}/curves/{tag}_fppi_mr_iou_{iou}_{subset_name}.pdf',
                               bbox_inches='tight')
                fig_pr.savefig(f'{root_path}/{tag}_eval_{pth_name}/curves/{tag}_pr_iou_{iou}_{subset_name}.pdf',
                               bbox_inches='tight')
            print(f'\ncurves have been saved!')
        else:

            # test if the model trained with the new loss term
            # if yes, starting the evaluation with the modified post-processing
            if int(contra_weighting_factor) != 0:
                print(f'detected using new loss, now evaluate with the modified post-processing approach...')
                print(f"Params: b: {cfg['b']}, c: {cfg['c']}, bias: {cfg['bias']}")

                # print('val-set')
                # # reload the model w/o confidence threshold
                # model_zero_conf, _ = load_model(cfg, zero_conf_thres=True)
                #
                # # register hooks on the feature maps
                # feature_maps = []
                # handle = model_zero_conf.backbone.register_forward_hook(lambda module, input,
                #                                                                output: feature_maps.append(output))
                # # Evaluates the model weights from the checkpoint file on the test set
                # results_score_boost = evaluate_model(model_zero_conf, val_loader, device,
                #                                      conf_threshold_post_pros=float(standard_conf_thres[cfg['model_name']]),
                #                                      feat_maps=feature_maps)
                # handle.remove()
                # # save the eval results into a json file
                # with open(f'{root_path}/{tag}_eval_{pth_name}/{tag}_eval_result_score_boosting_val_set.json', 'w') as fp:
                #     json.dump(results_score_boost, fp)
                #
                # del model_zero_conf, results_score_boost, feature_maps  # free space

                # load the model w/o confidence threshold
                model_zero_conf, _ = load_model(cfg, zero_conf_thres=True)

                # register hooks on the feature maps
                feature_maps = []
                handle = model_zero_conf.backbone.register_forward_hook(lambda module, input,
                                                                               output: feature_maps.append(output))
                # Evaluates the model weights from the checkpoint file on the test set
                results, scores = evaluate_model(model_zero_conf, dataset, device,
                                                     conf_threshold_post_pros=float(standard_conf_thres[cfg['model_name']]),
                                                     feat_maps=feature_maps, cfg=cfg)
                handle.remove()
                # save the eval results into a json file
                with open(f'{root_path}/{tag}_eval_{pth_name}/{tag}_eval_result_score_boosting_{subset_name}.json', 'w') as fp:
                    json.dump(results, fp)
                #with open(f'{root_path}/{tag}_eval_{pth_name}/{tag}_scores_{subset_name}.csv', 'wb') as f:
                #    w = csv.writer(f)
                #    w.writerow(scores.keys())
                #    w.writerow(scores.values())

                df=pd.DataFrame(columns=['before_tp', 'before_fp', 'after_tp', 'after_fp'])
                df.loc[0]=[
                    scores['before_tp'],
                    scores['before_fp'],
                    scores['after_tp'],
                    scores['after_fp'],
                           ]
                print(f'{root_path}/{tag}_eval_{pth_name}/{tag}_scores_{subset_name}.p')
                df.to_pickle(f'{root_path}/{tag}_eval_{pth_name}/{tag}_scores_{subset_name}.p')

                #with open(f'{root_path}/{tag}_eval_{pth_name}/{tag}_scores_{subset_name}.json', 'w') as sc:
                #    json.dumps(str(scores), sc)
                fig_scores = plot_score_distributions(scores)
                fig_scores.savefig(f'{root_path}/{tag}_eval_{pth_name}/curves/{tag}_score_distro.png',
                                   bbox_inches='tight')

                del model_zero_conf,  feature_maps

                print(f'\nscore boosting eval results have been saved!')
        #--------- AUC Precision Recall Analysis ---------------
        roc_results=results['roc']
        fig_roc = plot_AUC_precision_recall(roc_results)
        fig_roc.savefig(f'{root_path}/{tag}_eval_{pth_name}/curves/{tag}_roc_prec_recall_curve_mode={cfg["mode"]}.pdf',
                 bbox_inches='tight')
        print(f"mode: {cfg['mode']}, b: {cfg['b']}, c:{cfg['c']}, bias: {cfg['bias']}")
          # free space
    # ------------------------- part 2: inference images -------------------------
    if args.save_vis:
        if not os.path.exists(f'{root_path}/{tag}_eval_{pth_name}/example_images_results'):
            os.makedirs(f'{root_path}/{tag}_eval_{pth_name}/example_images_results')

        print(f'\nsaving example images...')
        feature_maps_visul = []  # register hooks on the feature maps
        handle = model.backbone.register_forward_hook(lambda module, input, output: feature_maps_visul.append(output))
        # TODO: 1 - heatmap for some baseline models, do it in a func by visual inspecting
        visul_heatmap(model, device, dataset, feature_maps_visul, tag=tag,
                      saving_root=f'{root_path}/{tag}_eval_{pth_name}/example_images_results',
                      conf_thres=0.25, n_imgs=10)
        handle.remove()
        print(f'\nexample images saved!')

    # # ------------------------- part 3: PLF factors extraction -------------------------
    if args.plfs:
        print(f'\nsaving PLFs dataframe...')
        # extract PLFs dataframe
        conf_threshold_list = np.arange(0, 101, 5) / 100  # 21 confidence thresholds
        saving_name = f'{tag}_{pth_name}_{subset_name}'
        extract_plf(
            model, device, dataset, cfg['pretrained_weights_path'],
            conf_thr_list=conf_threshold_list,
            saving_name=saving_name,
            iou_thr=0.25,
        )
        print('\nPLFs dataframe saved!')


if __name__ == '__main__':


    main()
