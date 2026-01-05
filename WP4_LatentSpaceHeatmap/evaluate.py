import time
import numpy as np
import torch
import utils


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
        if outputs[batch_idx]['boxes'].nelement() != 0:
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
    model.to(device)
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