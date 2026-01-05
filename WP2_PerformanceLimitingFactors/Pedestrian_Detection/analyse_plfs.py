import argparse
import os.path
import pickle

import fiftyone as fo
from fiftyone import ViewField as F
import numpy as np
import yaml
import cv2


def load_fo_dataset(dataset_name, prediction_fields, conf_thresholds):
    """
    Loads a FiftyOne dataset instance. The model predictions within the dataset are thresholded using the model specific
    confidence thresholds. Furthermore, a dataset view containing only the safety relevant ground truths is created
    and returned as another instance.

    :param dataset_name: The string name of the dataset, which should be used for the analysis (Options:
    'KIA-Pedestrian-Detection' or 'CityPersons').
    :param prediction_fields: Dictionary, which maps the names of the detection models to the corresponding FiftyOne
    label field names.
    :param conf_thresholds: Dictionary, which holds the information about the model specific confidence thresholds for
    each dataset and model.
    :return: A FiftyOne dataset instance with thresholded predictions based on the model specific confidence thresholds,
    and a dataset view, which contains only the safety relevant ground truths.
    """
    dataset = fo.load_dataset(dataset_name)  # Loads the FiftyOne dataset instance
    # Filters out low confidence detections made by the models, based on the model specific confidence thresholds
    for model in prediction_fields:
        dataset = dataset.filter_labels(
            prediction_fields[model],
            F('confidence') >= conf_thresholds[dataset_name][model],
            only_matches=False
        )
    # Filters out all non-safety relevant ground truth labels from the dataset and stores the reference within a new
    # variable
    sr_dataset = dataset.filter_labels(
        'ground_truth',
        F('ignore') == False,
        only_matches=False
    )
    print(dataset)  # Logs the dataset to the console
    return dataset, sr_dataset


def get_num_plf_bounds(plfs, dataset, sr_dataset):
    """
    Computes the minimum and maximum values for each numerical PLF within a given FiftyOne dataset instance.

    :param plfs: Dictionary, which holds the names of the performance limiting factors (PLFs), which appear as FiftyOne
    sample/detection attributes and are categorized based on whether they are computed for a sample or an object, and
    whether they are numerical or categorical.
    :param dataset: A FiftyOne dataset instance.
    :param sr_dataset: A view of a FiftyOne dataset instance, containing only safety relevant ground truths.
    :return: A dictionary, which contains the plf computation source as the string keys ('sample' and 'object'). Each
    of these keys contains another dictionary, with the string names of the PLFs as the keys. Each of these PLF keys
    contains a tuple as its value, where the first item is the respective minimum and the second item is the respective
    maximum value of that PLF within the dataset.
    """
    # Used for tracking the min and max values of each numerical PLF, which will be later used for normalization
    num_plf_bounds = {
        'sample': {},
        'object': {}
    }
    # Iterates over all computation sources ('sample' or 'object')
    for plf_comp_source in plfs:
        # Iterates over all numerical PLFs
        for plf in plfs[plf_comp_source]['numerical']:
            if plf_comp_source == 'sample':
                # Stores the min and max values for the sample PLF
                num_plf_bounds[plf_comp_source][plf] = dataset.bounds(plf, safe=True)
            elif plf_comp_source == 'object':
                # Stores the min and max values for the object PLF (only safety relevant pedestrians considered)
                num_plf_bounds[plf_comp_source][plf] = sr_dataset.bounds(f'ground_truth.detections.{plf}', safe=True)
    print(num_plf_bounds)  # Logs the minimum and maximum values for each numerical PLF to the console
    return num_plf_bounds


def get_plf_performance_ds(plfs, dataset, prediction_fields, cfg):
    """
    Loads the datastructure for tracking the detection performance for each model, with respect to the PLF values.

    :param plfs: Dictionary, which holds the names of the performance limiting factors (PLFs), which appear as FiftyOne
    sample/detection attributes and are categorized based on whether they are computed for a sample or an object, and
    whether they are numerical or categorical.
    :param dataset: A FiftyOne dataset instance.
    :param prediction_fields: Dictionary, which maps the names of the detection models to the corresponding FiftyOne
    label field names.
    :return: A dictionary, which tracks the detection performance of each model with respect to the PLF values.
    """
    # Used for tracking the detection performance of each model w.r. to the PLF values
    plf_performance = {
        model: {
            plf_comp_source: {
                plf_type: {
                    plf: {} for plf in plfs[plf_comp_source][plf_type]  # Iterates over all PLFs
                } for plf_type in plfs[plf_comp_source]  # Iterates over all PLF types ('numerical' or 'categorical')
            } for plf_comp_source in plfs  # Iterates over all computation sources ('sample' or 'object')
        } for model in prediction_fields  # Iterates over all models
    }

    # Iterates over all models
    for model in plf_performance:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance[model][plf_comp_source][plf_type]:

                    # Handles numerical PLFs
                    if plf_type == 'numerical':
                        # Numerical factors are being tracked using 100 buckets for the values
                        # (range from 0 to 1 with a step size of 0.01)
                        plf_performance[model][plf_comp_source][plf_type][plf] = {
                            'tps': np.array([0. for _ in range(101)]),
                            'srtps': np.array([0. for _ in range(101)]),
                            'fns': np.array([0. for _ in range(101)]),
                            'recall': np.array([0. for _ in range(101)])
                        }
                        if plf_comp_source == 'sample':
                            # Only sample PLFs are tracking the amount of false positives, which enables the computation
                            # of the precision and F1-score
                            plf_performance[model][plf_comp_source][plf_type][plf]['fps'] = np.array(
                                [0. for _ in range(101)]
                            )
                            plf_performance[model][plf_comp_source][plf_type][plf]['precision'] = np.array(
                                [0. for _ in range(101)]
                            )
                            plf_performance[model][plf_comp_source][plf_type][plf]['F1'] = np.array(
                                [0. for _ in range(101)]
                            )
                        if cfg['evaluate_heatmaps']:
                            plf_performance[model][plf_comp_source][plf_type][plf]['tp_hs']=np.array([0. for _ in range(101)])
                            plf_performance[model][plf_comp_source][plf_type][plf]['fn_hs']=np.array([0. for _ in range(101)])
                            plf_performance[model][plf_comp_source][plf_type][plf]['recall_h']=np.array([0. for _ in range(101)])

                    # Handles categorical PLFs
                    elif plf_type == 'categorical':
                        if plf_comp_source == 'sample':
                            # Determines all available sample categories for a specific sample PLF
                            plf_categories = dataset.distinct(plf)
                        elif plf_comp_source == 'object':
                            # Determines all available object categories for a specific object PLF
                            plf_categories = dataset.distinct(f'ground_truth.detections.{plf}')
                        # Categorical PLFs are being tracked using single buckets for each category
                        plf_performance[model][plf_comp_source][plf_type][plf] = {
                            'tps': {cat: 0. for cat in plf_categories},
                            'srtps': {cat: 0. for cat in plf_categories},
                            'fns': {cat: 0. for cat in plf_categories},
                            'recall': {cat: 0. for cat in plf_categories}
                        }
                        if plf_comp_source == 'sample':
                            # Only sample PLFs are tracking the amount of false positives, which enables the computation
                            # of the precision and F1-score
                            plf_performance[model][plf_comp_source][plf_type][plf]['fps'] = {
                                cat: 0. for cat in plf_categories
                            }
                            plf_performance[model][plf_comp_source][plf_type][plf]['precision'] = {
                                cat: 0. for cat in plf_categories
                            }
                            plf_performance[model][plf_comp_source][plf_type][plf]['F1'] = {
                                cat: 0. for cat in plf_categories
                            }
                        if cfg['evaluate_heatmaps']:
                            plf_performance[model][plf_comp_source][plf_type][plf]['tp_hs']={cat: 0. for cat in plf_categories}
                            plf_performance[model][plf_comp_source][plf_type][plf]['fn_hs']={cat: 0. for cat in plf_categories}
                            plf_performance[model][plf_comp_source][plf_type][plf]['recall_h']={cat: 0. for cat in plf_categories}
    return plf_performance


def get_plf_values(fo_instance, plf_comp_source, plfs, num_plf_bounds, plf_values=None):
    """
    This functions uses a FiftyOne sample/detection instance (fo_instance), whose computation source is specified by the
    plf_comp_source argument, to extract the corresponding values for the PLFs. The values are either stored within a
    given dictionary (plf_values) or a new dictionary is generated within this function.

    :param fo_instance: A FiftyOne sample/detection instance.
    :param plf_comp_source: A string that specifies the computation source for the PLF ('sample' or 'object').
    :param plfs: Dictionary, which holds the names of the performance limiting factors (PLFs), which appear as FiftyOne
    sample/detection attributes and are categorized based on whether they are computed for a sample or an object, and
    whether they are numerical or categorical.
    :param num_plf_bounds: A dictionary, which contains the minimum and maximum occurrence for each numerical PLF
    within the dataset. This information is used for normalizing the raw numerical PLF values.
    :param plf_values: A dictionary object, which stores all the PLF values for a specific object and the corresponding
    sample it belongs to.
    :return: A dictionary, which contains 2 string keys ('sample' and 'object') specifying the PLF computation source.
    Each of these keys contains another dictionary as the value, where the keys of that dictionary correspond to the
    PLF type ('numerical' and 'categorical'). The values for each of these keys is another dictionary that maps the
    string name of the PLF to the value it has.
    """
    if not plf_values:
        # Dictionary for storing the PLF values for a specific object and the corresponding sample it belongs to
        plf_values = {
            plf_comp_source: {
                plf_type: {
                    plf: {} for plf in plfs[plf_comp_source][plf_type]
                } for plf_type in plfs[plf_comp_source]
            } for plf_comp_source in plfs
        }

    # Iterates over all PLF types ('numerical' or 'categorical')
    for plf_type in plfs[plf_comp_source]:
        # Iterates over all PLFs
        for plf in plfs[plf_comp_source][plf_type]:
            if plf_type == 'numerical':
                # Extracts the numerical PLF value and normalizes it
                min_val = num_plf_bounds[plf_comp_source][plf][0]
                max_val = num_plf_bounds[plf_comp_source][plf][1]
                raw_val = fo_instance[plf]
                if raw_val is None or np.isnan(raw_val):
                    plf_values[plf_comp_source][plf_type][plf] = {}
                    continue  # Skips None and NaN values
                plf_values[plf_comp_source][plf_type][plf] = normalize(raw_val, min_val, max_val)
            elif plf_type == 'categorical':
                category = fo_instance[plf]
                if category is None:
                    plf_values[plf_comp_source][plf_type][plf] = {}
                    continue  # Skips None values
                # Stores the category of the categorical PLF
                plf_values[plf_comp_source][plf_type][plf] = category
    return plf_values


def store_plf_performance(plf_performance, plf_values, model, eval_state):
    """
    This function is responsible for tracking the detection performance (which is given as a string within the
    eval_state argument) with respect to the values of the PLFs (which are stored within the plf_values dictionary). It
    counts the occurrences of tps, srtps, fns and fps within the plf_performance data structure, for each model and for
    each PLF value bucket.

    :param plf_performance: A dictionary for tracking the detection performance for each model with respect to the
    PLF values.
    :param plf_values: A dictionary, which contains 2 string keys ('sample' and 'object') specifying the PLF computation
    source. Each of these keys contains another dictionary as the value, where the keys of that dictionary correspond to
    the PLF type ('numerical' and 'categorical'). The values for each of these keys is another dictionary that maps the
    string name of the PLF to the value it has.
    :param model: String name of the model for which the detection performance is being updated.
    :param eval_state: A string that specifies the evaluation state of the detection ('tp', 'srtp', 'fp' or 'fn').
    :return: Returns the plf_performance dictionary with updated detection performance values.
    """
    # Iterates over all computation sources ('sample' or 'object')
    for plf_comp_source in plf_values:
        if plf_comp_source == 'object':
            if eval_state == 'fp' or eval_state == 'tp':
                continue  # Only safety relevant pedestrians ('srtp' and 'fn') are considered for object PLFs
        # Iterates over all PLF types ('numerical' or 'categorical')
        for plf_type in plf_values[plf_comp_source]:
            # Iterates over all PLFs
            for plf in plf_values[plf_comp_source][plf_type]:
                plf_value = plf_values[plf_comp_source][plf_type][plf]  # Extracts the value of the PLF
                if plf_value == {}:
                    continue  # Skips non-existing PLF values (values that were None or NaN)
                # Increases the eval_state counter w.r. to the model and PLF value bucket
                if plf_type == 'numerical':
                    plf_performance[model][plf_comp_source][plf_type][plf][eval_state+'s'][round(plf_value * 100)] += 1
                elif plf_type == 'categorical':
                    plf_performance[model][plf_comp_source][plf_type][plf][eval_state+'s'][plf_value] += 1
    return plf_performance


def normalize(value, min_val, max_val):
    """
    Normalizes a given numerical value by subtracting the minimum occurrence of that value and dividing by the
    subtraction of the maximum and minimum occurrence of that value.

    :param value: A numerical value that should be normalized.
    :param min_val: The minimum occurrence of that value for normalization.
    :param max_val: The maximum occurrence of that value for normalization.
    :return: The normalized value.
    """
    return (value - min_val) / (max_val - min_val)


def get_precision(tps, fps):
    """
    Computes the precision value from the amount of true positives and false positives.

    :param tps: An integer that specifies the amount of true positives.
    :param fps: An integer that specifies the amount of false positives.
    :return: The precision value.
    """
    try:
        return tps / (tps + fps)
    except Exception as e:
        return float('NaN')


def get_recall(srtps, fns):
    """
    Computes the recall value from the amount of safety relevant true positives and false negatives.

    :param srtps: An integer that specifies the amount of safety relevant true positives.
    :param fns: An integer that specifies the amount of false negatives.
    :return: The recall value.
    """
    try:
        return srtps / (srtps + fns)
    except Exception as e:
        return float('NaN')


def get_f1(precision, recall):
    """
    Computes the F1-score from the respective precision and recall values.

    :param precision: A float value that specifies the precision.
    :param recall: A float value that specifies the recall.
    :return: The F1-score.
    """
    try:
        return (2 * precision * recall) / (precision + recall)
    except Exception as e:
        return float('NaN')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/analyse_plfs.yaml',
                        help='Path to the config yaml file, which contains all the parameters for the PLF analysis.')
    args = parser.parse_args()

    # Reads the yaml file, which contains the parameters for the PLF analysis
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the FiftyOne dataset instance and a dataset view containing only safety relevant ground truths
    dataset, sr_dataset = load_fo_dataset(cfg['dataset_name'], cfg['prediction_fields'], cfg['conf_thresholds'])
    # Computes the bounds (minimum and maximum values within the dataset) for each numerical PLF
    if os.path.exists(f'/path/to//num_plf_bounds_{cfg["dataset_name"]}.p'):
        with open(f'/path/to//num_plf_bounds_{cfg["dataset_name"]}.p', 'rb') as file:
            num_plf_bounds=pickle.load(file)
    else:
        num_plf_bounds = get_num_plf_bounds(cfg['plfs'], dataset, sr_dataset)
        with open(f'/path/to//num_plf_bounds_{cfg["dataset_name"]}.p', 'wb') as file:
            pickle.dump(num_plf_bounds, file, protocol=pickle.HIGHEST_PROTOCOL)
    # Loads the datastructure for tracking the detection performance for each model with respect to the PLF values
    plf_performance = get_plf_performance_ds(cfg['plfs'], dataset, cfg['prediction_fields'], cfg)
    # Iterates over the whole dataset to aggregate the prediction performance with respect to the PLFs
    plf_performance = compute_plf_performance(
        plf_performance, cfg['plfs'], num_plf_bounds, dataset, cfg['prediction_fields'], cfg['eval_keys'], cfg
    )

    # Generates a new dictionary to hold the final results
    plf_analysis = {
        'plf_performance': plf_performance,
        'num_plf_bounds': num_plf_bounds
    }
    with open(f"{cfg['destination_path']}/{cfg['dataset_name']}_plf_analysis_heatmaps_25.p", 'wb') as handle:
        pickle.dump(plf_analysis, handle, protocol=pickle.HIGHEST_PROTOCOL)  # Stores the new dictionary

def get_mask_intersection(cfg, heatmap, bbox_rel):
    img_h, img_w=heatmap.shape
    bbox={
        'x1':int(bbox_rel[0]*img_w),
        'y1':int(bbox_rel[1]*img_h),
        'x2':int(bbox_rel[2]*img_w)+int(bbox_rel[0]*img_w),
        'y2':int(bbox_rel[3]*img_h)+int(bbox_rel[1]*img_h)
    }
    heatmap_masked=(heatmap>cfg['heatmap_thr']).astype(int)
    heatmap_masked=heatmap_masked[bbox['y1']:bbox['y2'], bbox['x1']:bbox['x2']]
    intersection=np.sum(heatmap_masked)/heatmap_masked.size
    return intersection
def compute_plf_performance(plf_performance, plfs, num_plf_bounds, dataset, prediction_fields, eval_keys, cfg):
    """
    Iterates over the whole dataset to aggregate the prediction performance with respect to the PLFs for each model.

    :param plf_performance: A dictionary for tracking the detection performance for each model with respect to the
    PLF values.
    :param plfs: Dictionary, which holds the names of the performance limiting factors (PLFs), which appear as FiftyOne
    sample/detection attributes and are categorized based on whether they are computed for a sample or an object, and
    whether they are numerical or categorical.
    :param num_plf_bounds: A dictionary, which contains the minimum and maximum occurrence for each numerical PLF
    within the dataset. This information is used for normalizing the raw numerical PLF values.
    :param dataset: A FiftyOne dataset instance.
    :param prediction_fields: Dictionary, which maps the names of the detection models to the corresponding FiftyOne
    label field names.
    :param eval_keys: Dictionary, which maps the names of the detection models to the corresponding FiftyOne evaluation
    keys.
    :return: Returns the plf_performance dictionary with the final detection performance values with respect to the PLFs.
    """
    # Visualizes the loading bar
    with fo.ProgressBar() as pb:
        # Iterates over all dataset samples
        for idx, sample in enumerate(pb(dataset)):
            if idx>=5000:
                break
            sample_plf_values = get_plf_values(sample, 'sample', plfs, num_plf_bounds)  # Extracts the sample PLF values

            # Iterates over all ground truth pedestrian objects
            for gt_pedestrian in sample.ground_truth.detections:
                # Adds the object PLF values to the dictionary
                plf_values = get_plf_values(gt_pedestrian, 'object', plfs, num_plf_bounds, sample_plf_values)
                # Iterates over all detection models, which have been specified
                for model in eval_keys:
                    # Handles true positive detections
                    if gt_pedestrian[eval_keys[model]] == 'tp':
                        store_plf_performance(plf_performance, plf_values, model, 'tp')
                        # Handles true positive detections of safety relevant pedestrians
                        if not gt_pedestrian['ignore']:
                            store_plf_performance(plf_performance, plf_values, model, 'srtp')
                    # Handles false negatives of safety relevant pedestrians
                    elif gt_pedestrian[eval_keys[model]] == 'fn' and not gt_pedestrian['ignore']:
                        store_plf_performance(plf_performance, plf_values, model, 'fn')

                    # Calculate a 'TP' or 'FN' of the heatmaps generated by the model
                    if cfg['evaluate_heatmaps'] and not gt_pedestrian['ignore']:
                        heatmap_path = f'{cfg["heatmap_path"]}/{cfg["dataset_name"]}/{model}/{sample.file_name}.png'
                        if not os.path.exists(heatmap_path):
                            continue
                        heatmap = cv2.imread(heatmap_path, cv2.IMREAD_UNCHANGED)
                        heat_intersection=get_mask_intersection(cfg, heatmap, gt_pedestrian.bounding_box)
                        if heat_intersection>=cfg['heatmap_intersection_thr']:
                            store_plf_performance(plf_performance, plf_values, model, 'tp_h')
                        else:
                            store_plf_performance(plf_performance, plf_values, model, 'fn_h')



            # Iterates over all detection models, which have been specified
            for model in prediction_fields:
                # Iterates over all model detections
                for dt_pedestrian in sample[prediction_fields[model]].detections:
                    # Handles false positive detections
                    if eval_keys[model] in dt_pedestrian._fields_ordered:
                        if dt_pedestrian[eval_keys[model]] == 'fp':
                            store_plf_performance(plf_performance, plf_values, model, 'fp')

    # Iterates over all specified detection models to compute the precision, recall and F1-score w.r. to the PLFs
    for model in plf_performance:
        # Iterates over all computation sources ('sample' or 'object')
        for plf_comp_source in plf_performance[model]:
            # Iterates over all PLF types ('numerical' or 'categorical')
            for plf_type in plf_performance[model][plf_comp_source]:
                # Iterates over all PLFs
                for plf in plf_performance[model][plf_comp_source][plf_type]:

                    # Computes the respective precision, recall and F1-score for numerical PLFs
                    if plf_type == 'numerical':
                        recall = get_recall(
                            plf_performance[model][plf_comp_source][plf_type][plf]['srtps'],
                            plf_performance[model][plf_comp_source][plf_type][plf]['fns']
                        )
                        plf_performance[model][plf_comp_source][plf_type][plf]['recall'] = recall
                        if cfg['evaluate_heatmaps']:
                            recall_h = get_recall(
                                plf_performance[model][plf_comp_source][plf_type][plf]['tp_hs'],
                                plf_performance[model][plf_comp_source][plf_type][plf]['fn_hs']
                            )
                            plf_performance[model][plf_comp_source][plf_type][plf]['recall_h'] = recall_h

                        if plf_comp_source == 'sample':
                            # Only sample PLFs are tracking the amount of false positives and can therefor
                            # compute the precision and F1-score
                            precision = get_precision(
                                plf_performance[model][plf_comp_source][plf_type][plf]['tps'],
                                plf_performance[model][plf_comp_source][plf_type][plf]['fps']
                            )
                            plf_performance[model][plf_comp_source][plf_type][plf]['precision'] = precision
                            f1 = get_f1(precision, recall)
                            plf_performance[model][plf_comp_source][plf_type][plf]['F1'] = f1

                    # Computes the respective precision, recall and F1-score for categorical PLFs
                    elif plf_type == 'categorical':
                        # Iterates over all categories for a specific categorical PLF
                        for cat in plf_performance[model][plf_comp_source][plf_type][plf]['tps']:
                            recall = get_recall(
                                plf_performance[model][plf_comp_source][plf_type][plf]['srtps'][cat],
                                plf_performance[model][plf_comp_source][plf_type][plf]['fns'][cat]
                            )
                            plf_performance[model][plf_comp_source][plf_type][plf]['recall'][cat] = recall
                            if cfg['evaluate_heatmaps']:
                                recall_h = get_recall(
                                    plf_performance[model][plf_comp_source][plf_type][plf]['tp_hs'][cat],
                                    plf_performance[model][plf_comp_source][plf_type][plf]['fn_hs'][cat]
                                )
                                plf_performance[model][plf_comp_source][plf_type][plf]['recall_h'][cat] = recall_h
                            if plf_comp_source == 'sample':
                                # Only sample PLFs are tracking the amount of false positives and can therefor
                                # compute the precision and F1-score
                                precision = get_precision(
                                    plf_performance[model][plf_comp_source][plf_type][plf]['tps'][cat],
                                    plf_performance[model][plf_comp_source][plf_type][plf]['fps'][cat]
                                )
                                plf_performance[model][plf_comp_source][plf_type][plf]['precision'][cat] = precision
                                f1 = get_f1(precision, recall)
                                plf_performance[model][plf_comp_source][plf_type][plf]['F1'][cat] = f1
    return plf_performance


if __name__ == '__main__':
    main()
