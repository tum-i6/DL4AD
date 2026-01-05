import fiftyone as fo
from fiftyone import ViewField as F
import pandas as ps
import argparse
import yaml

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
    # Filters out low confidence detections made by the models, based on the model specific confidence thresholds, which are specified in the configurations
    for model in cfg['prediction_fields']:
        dataset = dataset.filter_labels(
            cfg['prediction_fields'][model], F('confidence') >= cfg['conf_thresholds'][cfg['dataset_name']][model],
            only_matches=False
        )

    # Filters out all non-safety relevant ground truth labels from the dataset and stores the reference within a new variable
    sr_dataset = dataset.filter_labels('ground_truth', F('ignore') == False, only_matches=False)

if __name__ == '__main__':
    main()