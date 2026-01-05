import time
import pickle
import random
import numpy as np
import torchvision.transforms
import torch
import yaml

import utils
from evaluate import evaluate_one_batch
from extract_activation_patches import standard_kia_split, get_transform, ToTensor, Compose, collate_fn, \
    RandomHorizontalFlip, KIA_dataset, dts_gts_match, load_trained_faster_rcnn


def testset_dataloader_kia(cfg):
    """
    Loads the test dataloader instances for the KIA dataset.

    :param cfg: A configuration dictionary containing the following parameter values: 'seed' is an integer number that
    should be used as the seed for the random module in order to support reproducibility, 'model_name' is the string
    name of the detection model, 'data_path' contains the string path to the dataset root folder, 'min_obj_pixels'
    specifies the visible pixel amount threshold for filtering pedestrians, 'max_occl' and 'min_occl' specify the occlusion
    threshold for filtering pedestrians, 'max_distance' specifies the distance threshold for filtering pedestrians,
    'batch_size' specifies the amount of image samples in a single training batch and 'num_workers' specifies the amount
    of sub-processes for loading the data.
    :return: A test dataloader instance, containing samples from the KIA dataset.
    """
    # Sets the random seed for reproducibility
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    # Checks whether instance segmentation or keypoint labels should be loaded
    instance_segm = 'IS' in cfg['model_name']
    keypoint_det = 'KD' in cfg['model_name']
    # Loads the KIA dataset instances for training, validation and testing of the detection model
    train_dataset = KIA_dataset(
        cfg['data_path'],
        get_transform(train=False),
        min_obj_pixels=cfg['min_obj_pixels'],
        max_occl=cfg['max_occl'],
        min_occl=cfg['min_occl'],
        max_distance=cfg['max_distance'],
        instance_segm=instance_segm,
        keypoint_detect=keypoint_det
    )
    test_dataset = KIA_dataset(root_path=cfg['data_path'],
                               min_obj_pixels=cfg['min_obj_pixels'],
                               max_occl=cfg['max_occl'],
                               min_occl=cfg['min_occl'],
                               max_distance=cfg['max_distance'])

    # Used for storing the sample indices for the train, validation and test samples, based on the standard KIA split
    train_indices = []
    val_indices = []
    test_indices = []

    # Iterates over all dataset samples to determine whether they belong to the train, validation or test split
    for sample_idx, img_sample_path in enumerate(
            train_dataset.imgs):  # f'{tranche}/{sequence}/'sensor/camera/left/png'/{file_name}.png'
        sample_info = img_sample_path.split('/')
        company_name = sample_info[0].split('_')[2]
        tranche = sample_info[0].split('_')[-1]
        sequence = int(sample_info[1].split('_')[3].split('-')[0].lstrip(
            '0'))  # A set of characters to remove as leading characters
        if sequence in standard_kia_split['train_set'][company_name][tranche]:
            train_indices.append(sample_idx)
        elif sequence in standard_kia_split['val_set'][company_name][tranche]:
            val_indices.append(sample_idx)
        elif sequence in standard_kia_split['test_set'][company_name][tranche]:
            test_indices.append(sample_idx)

    # Splits the dataset samples into test samples
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
    min_occl = cfg['min_occl']
    max_occl = cfg['max_occl']
    occlusion_range = f'{int(100*min_occl)}-{int(100*max_occl)}'
    if occlusion_range in cfg['indices']:
        test_dataset = torch.utils.data.Subset(test_dataset, cfg['indices'][occlusion_range])
    else:
        print('The given occlusion range is not provided, pls choose from the following 4 ranges\n')
        print('15-25\n35-45\n55-65\n75-85')
    # Loads the test dataset into PyTorch dataloader instances
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg['num_workers'],
        collate_fn=collate_fn
    )
    return test_loader


if __name__ == '__main__':
    # Reads the yaml file, which contains the parameters for loading the training dataset
    cfg_path = './config/testset_outlier_detector.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # load the dataloader for KIA training set
    test_loader = testset_dataloader_kia(cfg)
    print(len(test_loader))
    # load the pre-trained model
    weight_path = './model_weights/final_model.pth'
    model, device = load_trained_faster_rcnn(weight_path)
    model.eval()  # set the model to eval mode
    # send model to devices
    model.to(device)
    # load the outlier detector
    filename = 'isolation_forest_model.sav'
    # load the model from disk
    outlier_model = pickle.load(open(filename, 'rb'))
    # a list to store all the last feature maps for all the samples in the training dataloader
    feature_maps = []
    # register the forward hook
    model.backbone.body.layer4[2].conv3.register_forward_hook(lambda module, input, output: feature_maps.append(output))

    # specify the cpu device
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
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
    # for calculating AP of outlier detector
    n_samples = 0
    n_tp = 0
    n_fn = 0

    # process the input image in training dataset one by one
    for image, target in metric_logger.log_every(test_loader, 100, "Test:"):
        # send the input image to GPU
        image = [img.to(device) for img in image]
        with torch.no_grad():
            # get the width and height of the input image
            c_img, h_img, w_img = image[0].size()  # [C, H, W]
            # Synchronizes the GPU to measure the inference time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            model_time = time.time()
            # feed the input image to model
            outputs = model(image)
            # move the results to cpu
            outputs_cpu = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            # keep the detections which have the largest iou with the zero-occluded pedestrian ground truth bboxes
            outputs_matched = dts_gts_match(outputs_cpu, target, score_thresh=0.25, iou_thresh=0.5)
            # test if there is any matched prediction
            # if len(outputs_matched) == 0:
            #     continue  # move to the next input image
            model_time = time.time() - model_time  # Tracks the model's inference time
            n_samples += len(target[0]['boxes'])
            if outputs_matched[0]['boxes'].nelement() != 0:
                # get the corresponding feature map regarding the particular input image
                feature_map = feature_maps.pop()[0].clone()
                # get the width and height of the feature map
                c_f, h_f, w_f = feature_map.shape  # with shape [C, H, W]
                # transform factors between bboxes and feature map
                w_factor, h_factor = w_f / w_img, h_f / h_img
                # apply the transform factor to the prediction bboxes
                refactored_bboxes = outputs_matched[0]['boxes'].clone()
                refactored_bboxes[:, [0, 2]] *= w_factor
                refactored_bboxes[:, [1, 3]] *= h_factor

                # turn the float coords into the integer coords
                refactored_bboxes[:, [0, 1]] = torch.floor(refactored_bboxes[:, [0, 1]])  # x1, x2 are floored
                refactored_bboxes[:, [2, 3]] = torch.ceil(refactored_bboxes[:, [2, 3]])  # x2, y2 are ceiled

                # extract the activation within the areas of refactored bboxes
                extracted_feature_maps = []
                for refactored_bbox in refactored_bboxes.int():
                    # extract the corresponding pieces of feature maps
                    extracted_feature_map = feature_map[:, refactored_bbox[1]:min(refactored_bbox[3], h_f),
                                            refactored_bbox[0]:min(refactored_bbox[2], w_f)]  # shape [C, H_e, W_e]
                    resized_tensor = torchvision.transforms.Resize(size=(1, 2))(
                        extracted_feature_map).cpu().numpy()
                    # append the extracted activation
                    extracted_feature_maps.append(resized_tensor)
                extraction_resized = np.array(extracted_feature_maps).reshape(len(extracted_feature_maps), -1)
                outlier_results = outlier_model.predict(extraction_resized)
                # calculate the number of samples, number of true positive and number of false negative
                n_tp += outlier_results[outlier_results == 1].size
                n_fn += outlier_results[outlier_results == -1].size

            evaluator_time = time.time()
            batch_eval_results = evaluate_one_batch(outputs_matched, target)  # Evaluates the batch of predictions
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
            fppi = fps / len(test_loader.dataset)

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
    print(
        f'Miss Rate                   (MR)    @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["MR"], 3)}')
    print(
        f'False Positives per Image   (FPPI)  @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["FPPI"], 3)}')
    print(
        f'F1 Score                    (F1)    @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["F1"], 3)}')
    print(
        f'Precision                   (P)     @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["precision"], 3)}')
    print(
        f'Recall                      (R)     @[ IoU=0.25           &  C=0.5          ] = {round(eval_results["0.5"]["0.25"]["recall"], 3)}')
    print(f'\nrecall of the outlier detector: {n_tp}/{n_samples} = {round(n_tp / n_samples * 100, 3)}%')

    torch.set_num_threads(n_threads)  # Sets the number of CPU threads as it was before the evaluation