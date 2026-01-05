import argparse

import os

import cv2
import torch
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluation import evaluate_one_batch
from roi_plf_extractor import extract_plf
from visul_feat_map import visul_heatmap
from models import load_model, standard_conf_thres
from data import load_dataloaders
import warnings
import time
import numpy as np
import utils
import seaborn as sns
import json
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def boost_scores(prob_map, output, b, c, conf_threshold=0.05):
    copied_output = output.copy()

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
            averaged_score = score.item() / (1 + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - c) * b))  # (5
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


def boost_scores_offline(prob_map, output, device, b, c, conf_threshold=0.05):
    copied_output = output.copy()
    # convert boxes to integer type
    output['boxes'] = output['boxes'].type(torch.int64)

    adjusted_scores = []

    for origi_box, box, score in zip(copied_output['boxes'], output['boxes'], output['scores']):
        x1, y1, x2, y2 = tuple(box.numpy())

        area = (x2 - x1) * (y2 - y1)
        if area == 0:
            averaged_score = score.item()
        else:

            averaged_score = score.item() / (1 + np.exp(-(np.mean(prob_map[y1:y2, x1:x2]) - c) * b))  # (5
            averaged_score = max(0, min(averaged_score, 1))

        adjusted_scores.append(averaged_score)

    copied_output['scores'] = torch.tensor(adjusted_scores, dtype=output['scores'].dtype)

    copied_output = {k: v[copied_output['scores'] > conf_threshold] for k, v in copied_output.items()}

    return copied_output
@torch.inference_mode()
def evaluate_model_grid_search(model, data_loader, device, b, c, conf_threshold_post_pros=None, feat_maps=None, dataframe=None, cfg=None):
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
    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, "Train:")):
        if cfg['dataset'] != 'CityPersons':
            if batch_idx > 1000:
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
                target = targets[0]
                output = outputs[0]
                # original input image
                original_h, original_w = img.size()[-2:]
                i_h, i_w = tuple([original_h, original_w])

                # create masks for each activation levels
                mask = torch.zeros_like(img.cpu())
                for gt in target['boxes'].clone():
                    x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
                    mask[:, y1: y2, x1: x2] = 1

                # channel-wise average & resize to input image space
                if cfg['model_name'] != 'SSD300-ResNet50-OD':
                    heatmap = torch.mean(activations['0'], 1).squeeze(0).numpy()
                else:
                    heatmap = torch.mean(activations, 1).squeeze(0).numpy()
                resized_heatmap = cv2.resize(heatmap, (i_w, i_h))

                # probability/ likelihood map with value 0-1 by sigmoid func
                p_map = 1 / (1 + np.exp(-resized_heatmap * 2))

                # TODO: let's try something new
                # p_map = np.heaviside(resized_heatmap, 0.5)  # step func

                # update the scores
                assert conf_threshold_post_pros is not None

                outputs = [boost_scores(p_map, output, b, c, conf_threshold=conf_threshold_post_pros)]

                dataframe.loc[batch_idx] = pd.Series({
                    'sample_id':batch_idx,
                    'label':target,
                    'img':img.cpu().numpy(),
                    'preds':output,
                    'heatmap':p_map
                })

        targets = [{k: v for k, v in t.items() if k != 'masks'} for t in targets]

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
    print('Calculating the metrics....')
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
    custom_ap = round(aps['0.25'], 3)  # AP with 0.25 IoU threshold
    custom_lamr = round(utils.get_log_avg_miss_rate(eval_results, '0.25'), 3)  # LAMR with 0.25 IoU threshold

    mean_recall = 0
    mean_precision = 0
    for conf in list(range(0, 101, 5)):
        conf /= 100
        mean_recall += eval_results[str(conf)]["0.25"]["recall"]
        mean_precision += eval_results[str(conf)]["0.25"]["precision"]
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
    }

    torch.set_num_threads(n_threads)  # Sets the number of CPU threads as it was before the evaluation
    return performance_summary, dataframe
def evaluate_model_grid_search_offline(model, data_loader, device, b, c, conf_threshold_post_pros=None, feat_maps=None, dataframe=None):
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
    for index, row  in tqdm(dataframe.iterrows()):
        img = row['img']
        target = row['label']
        output = row['preds']
        # original input image

        original_h=img.shape[1]
        original_w = img.shape[2]
        i_h, i_w = tuple([original_h, original_w])

        # create masks for each activation levels
        #mask = torch.zeros_like(img.cpu())
        #for gt in target['boxes'].clone():
        #    x1, y1, x2, y2 = tuple(gt.cpu().numpy().astype(int))
        #    mask[:, y1: y2, x1: x2] = 1

        # probability/ likelihood map with value 0-1 by sigmoid func
        p_map = row['heatmap']

        # TODO: let's try something new
        # p_map = np.heaviside(resized_heatmap, 0.5)  # step func

        # update the scores
        assert conf_threshold_post_pros is not None

        outputs = [boost_scores_offline(p_map, output,device, b, c, conf_threshold=conf_threshold_post_pros)]


        targets = [{k: v for k, v in t.items() if k != 'masks'} for t in [target]]


        #evaluator_time = time.time()
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
        #evaluator_time = time.time() - evaluator_time  # Tracks the time for computing the evaluation
        # Updates the information about the model's inference time and the time for computing the evaluation
        #metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # Logs the final stats about the model's inference time and the time for computing the evaluation to the console
    #print("Averaged stats:", metric_logger)

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
    custom_ap = round(aps['0.25'], 3)  # AP with 0.25 IoU threshold
    custom_lamr = round(utils.get_log_avg_miss_rate(eval_results, '0.25'), 3)  # LAMR with 0.25 IoU threshold

    mean_recall = 0
    mean_precision = 0
    for conf in list(range(0, 101, 5)):
        conf /= 100
        mean_recall += eval_results[str(conf)]["0.25"]["recall"]
        mean_precision += eval_results[str(conf)]["0.25"]["precision"]
    mean_recall /= len(range(0, 101, 5))
    mean_precision /= len(range(0, 101, 5))

    # Stores the computed metric values
    performance_summary = {
        'coco_ap': coco_ap,
        'pascal_voc_ap': pascal_voc_ap,
        'lamr': lamr,
        'custom_ap': custom_ap,
        'custom_lamr': custom_lamr,
        #'model_time': metric_logger.meters['model_time'].global_avg,
        #'evaluator_time': metric_logger.meters['evaluator_time'].global_avg,
        'eval_results': eval_results,
    }

    torch.set_num_threads(n_threads)  # Sets the number of CPU threads as it was before the evaluation
    return performance_summary

def main():
    # get config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='.config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    args = parser.parse_args()

    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    train_loader, val_loader, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # plot curves and save a couple of images
    root_path = '/'.join(cfg['pretrained_weights_path'].split('/')[:-1])
    contra_weighting_factor = int(root_path.split('/')[-1].split('_')[-1])  # in percentage

    model_name = cfg['model_name'].split('-')[0]  # e.g. FCOS
    dataset_name = cfg['dataset']  # e.g. CityPersons



    tag = f'{dataset_name}_{model_name}_wf_{contra_weighting_factor}'  # e.g. CityPersons_FCOS_wf_10
    print(tag)

    saving_path = f'/path/to/output_dir/grid_search_score_boosting/{tag}'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    if not os.path.exists(f'{saving_path}/train_dataframe.p'):
        if cfg['dataset']!='CityPersons':
            df_length=1000
        else:
            df_length = len(train_loader)
        df = pd.DataFrame(index=range(df_length), columns=[
            'sample_id',
            'label',
            'img',
            'preds',
            'heatmap'
        ])
    else:
        print('Loading the dataframe. Might take a while...')
        df=pd.read_pickle(f'{saving_path}/train_dataframe.p')
    if model_name == 'FasterRCNN' and dataset_name == "CityPersons":
        c_tuning = np.arange(0, 0.425, 0.025)
    else:
        c_tuning = np.arange(0, 1.05, .05)  # 21

    b_tuning = np.arange(1, 10.5, 0.5)  # 19

    coco_ap_matrix = np.zeros((len(c_tuning), len(b_tuning)))
    pascal_ap_matrix = np.zeros((len(c_tuning), len(b_tuning)))
    lamr_matrix = np.zeros((len(c_tuning), len(b_tuning)))

    precision_matrix = np.zeros((len(c_tuning), len(b_tuning)))
    recall_matrix = np.zeros((len(c_tuning), len(b_tuning)))

    if int(contra_weighting_factor) != 0:
        print(f'detected using new loss, now evaluate with the score boosting approach...')
        # reload the model w/o confidence threshold
        model_zero_conf, _ = load_model(cfg, zero_conf_thres=True)

        print('test-set')

        for idx_c, c in enumerate(c_tuning):
            for idx_b, b in enumerate(b_tuning):
                print(b, c)
                # register hooks on the feature maps
                feature_maps = []
                handle = model_zero_conf.backbone.register_forward_hook(lambda module, input,
                                                                               output: feature_maps.append(output))

                if not os.path.exists(f'{saving_path}/train_dataframe.p'):
                    # Evaluates the model weights from the checkpoint file on the train set
                    results_score_boost, df = evaluate_model_grid_search(model_zero_conf, train_loader, device, b=b, c=c,
                                                                     conf_threshold_post_pros=float(
                                                                         standard_conf_thres[cfg['model_name']]),
                                                                     feat_maps=feature_maps, dataframe=df, cfg=cfg)
                    print('saving the dataframe. Might take a while...')
                    df.to_pickle(f'{saving_path}/train_dataframe.p')
                else:
                    # Evaluates the model weights from the checkpoint file on the train set using saved dataframe
                    results_score_boost= evaluate_model_grid_search_offline(model_zero_conf, train_loader, device, b=b,
                                                                         c=c,
                                                                         conf_threshold_post_pros=float(
                                                                             standard_conf_thres[cfg['model_name']]),
                                                                         feat_maps=feature_maps, dataframe=df)

                handle.remove()
                del feature_maps

                coco_ap = results_score_boost['coco_ap']
                pascal_ap = results_score_boost['pascal_voc_ap']
                lamr = results_score_boost['lamr']

                precision = results_score_boost['eval_results']["0.5"]["0.25"]["precision"]
                recall = results_score_boost['eval_results']["0.5"]["0.25"]["recall"]

                coco_ap_matrix[idx_c, idx_b] = coco_ap
                pascal_ap_matrix[idx_c, idx_b] = pascal_ap
                lamr_matrix[idx_c, idx_b] = lamr

                precision_matrix[idx_c, idx_b] = round(precision, 3)
                recall_matrix[idx_c, idx_b] = round(recall, 3)

            # print(coco_ap_matrix)
            # print(recall_matrix)

    plt.subplots(figsize=(20, 15))
    ax0 = sns.heatmap(coco_ap_matrix, xticklabels=b_tuning, yticklabels=c_tuning, cmap="YlGnBu", annot=True, fmt='.3g')
    ax0.set_title(f'coco_ap_matrix\n{tag}')
    plt.xlabel('b')
    plt.ylabel('c')
    plt.savefig(f'{saving_path}/{tag}_coco_ap.png')
    plt.close()

    np.save(f'{saving_path}/{tag}_coco_ap_array.npy', coco_ap_matrix)

    plt.subplots(figsize=(20, 15))
    ax1 = sns.heatmap(pascal_ap_matrix, xticklabels=b_tuning, yticklabels=c_tuning, cmap="YlGnBu", annot=True, fmt='.3g')
    ax1.set_title(f'pascal_ap_matrix\n{tag}')
    plt.xlabel('b')
    plt.ylabel('c')
    plt.savefig(f'{saving_path}/{tag}_pascal_ap.png')
    plt.close()

    np.save(f'{saving_path}/{tag}_pascal_ap_array.npy', pascal_ap_matrix)

    plt.subplots(figsize=(20, 15))
    ax2 = sns.heatmap(lamr_matrix, xticklabels=b_tuning, yticklabels=c_tuning, cmap="YlGnBu", annot=True, fmt='.3g')
    ax2.set_title(f'lamr_matrix\n{tag}')
    plt.xlabel('b')
    plt.ylabel('c')
    plt.savefig(f'{saving_path}/{tag}_lamr.png')
    plt.close()

    np.save(f'{saving_path}/{tag}_lamr_array.npy', lamr_matrix)

    plt.subplots(figsize=(20, 15))
    ax3 = sns.heatmap(precision_matrix, xticklabels=b_tuning, yticklabels=c_tuning, cmap="YlGnBu", annot=True, fmt='.3g')
    ax3.set_title(f'precision_matrix\n{tag}')
    plt.xlabel('b')
    plt.ylabel('c')
    plt.savefig(f'{saving_path}/{tag}_precision.png')
    plt.close()

    np.save(f'{saving_path}/{tag}_precision_array.npy', precision_matrix)

    plt.subplots(figsize=(20, 15))
    ax4 = sns.heatmap(recall_matrix, xticklabels=b_tuning, yticklabels=c_tuning, cmap="YlGnBu", annot=True, fmt='.3g')
    ax4.set_title(f'recall_matrix\n{tag}')
    plt.xlabel('b')
    plt.ylabel('c')
    plt.savefig(f'{saving_path}/{tag}_recall.png')
    plt.close()

    np.save(f'{saving_path}/{tag}_recall_array.npy', recall_matrix)


if __name__ == '__main__':
    main()
