import argparse
import pickle
import os
import sys
import csv

from torchvision import transforms
from tqdm import trange
from PIL import Image
import numpy as np
import torch.nn as nn
import torch
import pandas
import yaml

import utils


def activate_model_dropout(model):
    """
    Activates all dropout layers inside the model by setting them to train mode.

    :param model: The classification model.
    """
    for layer in model.modules():
        if layer.__class__.__name__.startswith('Dropout'):
            layer.train()  # Set dropout layer to train mode


def apply_mc_dropout(model, sample, num_repeats):
    """
    Infers the sample image multiple times (num_repeats) into the model, which has dropout layers set to active. Returns
    the mean softmax score and the variance for each class based on all output scores for the given sample.

    :param model: The classification model used for inference.
    :param sample: Image sample used for inference.
    :param num_repeats: How many times to repeat the inference process.
    :return: The mean softmax score and the variance for each class based on all output scores for the given sample.
    """
    activate_model_dropout(model)  # Activates the dropout layers inside the model
    softmax = nn.Softmax(dim=1)
    output_stack = torch.zeros((num_repeats, 43))  # Placeholder for all output scores

    # Infers the image sample num_repeats times into the model and stacks the output scores onto the output_stack
    for i in range(num_repeats):
        with torch.no_grad():
            output = softmax(model(sample))
            output_stack[i] = output  # Stacks the output scores

    # Returns the mean output score and variance value for each class
    return torch.mean(output_stack, dim=0), torch.var(output_stack, dim=0)


def extract_attack_properties(attack_path, device):
    """
    Uses the attack path to extract all attack properties and creates the target output vectors for calculating the
    prediction loss.

    :param attack_path: A string of the attack path.
    :param device: The device used during the evaluation.
    :return: The name of the victim model (string), the name of the attack algorithm (string), the victim class
    (integer), attack target class (integer or False), target output vector (torch tensor) and attack target output
    vector (torch tensor or None)
    """
    attack_folder_name = attack_path.split('/')[-1]
    attack_properties = attack_folder_name.split('_')
    victim_model = attack_properties[2]
    attack_algorithm = attack_properties[3]
    victim_class = attack_properties[6]
    # Label tensor, which is used for calculating the prediction loss
    target_output = torch.tensor(int(victim_class)).type(torch.LongTensor).unsqueeze(0).to(device)
    if '_attack_target_class_' in attack_folder_name:
        attack_target_class = attack_properties[-1]
        # Label tensor, which is used for calculating the prediction loss for the attack target class
        attack_target_output = torch.tensor(int(attack_target_class)).type(torch.LongTensor).unsqueeze(0).to(device)
    else:
        attack_target_class = False
        attack_target_output = None

    return victim_model, attack_algorithm, victim_class, attack_target_class, target_output, attack_target_output


def evaluate_attack(model, model_name, device, criterion, dataset, dataset_type, attack_path, csv_path,
                    overlay_file_name, extract_dataframe, target_layer, mc_dropout_num_repeats):
    """
    Evaluates the model's prediction performance on the attacked dataset samples.

    :param model: The classification model used for inference.
    :param model_name: The name of the classification model used for inference.
    :param device: The device used during the evaluation.
    :param criterion: The loss function used for calculating the prediction loss.
    :param dataset: A tuple containing two lists (the dataset samples and the dataset labels).
    :param dataset_type: The name of the dataset (train, valid, or test).
    :param attack_path: The path to the attack session folder created by realistic_sticker_attack.py.
    :param csv_path: The path for storing the final csv file, which will contain the evaluation results.
    :param overlay_file_name: The file name of the sticker overlay png file, which should be evaluated. If set to empty
    string, the evaluation will evaluate all sticker overlay files inside the sticker_overlays folder.
    :param extract_dataframe: Whether the evaluation should be used for extracting a pandas dataframe, which will
    contain different prediction properties for each evaluation sample.
    :param target_layer: Name of the model layer (attribute from the model class), from which to extract the model
    activations. Only used if extract_dataframe is set to True.
    :param mc_dropout_num_repeats: Amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only used
    if extract_dataframe is set to True.
    """
    # Extracts different attack properties from the attack's folder name, which is stored in the attack_path
    victim_model, attack_algorithm, victim_class, attack_target_class, target_output, attack_target_output = \
        extract_attack_properties(attack_path, device)

    if extract_dataframe:
        activation = {}  # Used for storing the target layer activations (feature maps)

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = getattr(model.module, target_layer)  # Extracts the given layer attribute from the model class
        # Extracts the feature maps from the target layer and stores them in activation
        layer.register_forward_hook(get_activation(target_layer))

        amount_samples = np.count_nonzero(dataset[1] == int(victim_class))
        # Defines the pandas dataframe, used for storing the model prediction properties
        df = pandas.DataFrame(index=range(amount_samples), columns=[
            'sample_id',
            'label',
            'img_attacked',
            'pred_attacked',
            'conf_attacked',
            'cscore_attacked',
            'layer_act_attacked',
            'cscore_mc_dropout_attacked',
            'cvariance_mc_dropout_attacked'
        ])

        softmax = nn.Softmax(dim=1)  # Applies softmax to the final output scores

    toTensor = transforms.ToTensor()  # Used for converting the image samples to torch tensors

    sticker_overlay_files = os.listdir(f"{attack_path}/sticker_overlays")  # Loads all sticker overlay file names

    # Iterates over all sticker overlays to evaluate their performance
    for sticker_overlay_file in sticker_overlay_files:

        # Checks whether the evaluation is set to evaluate only one sticker overlay file
        if overlay_file_name != '' and overlay_file_name != sticker_overlay_file:
            continue

        # Used for tracking the prediction performance for the clean and attacked images
        sample_counter = 0
        corrects_clean = 0
        corrects_attacked = 0
        corrects_target_attack = 0
        correct_attacked_mc_dropout_preds = 0  # Tracks the amount of correct predictions when using Monte Carlo Dropout
        clean_acc = 0
        attacked_acc = 0
        target_attack_acc = 0
        loss_clean = 0
        loss_attacked = 0
        loss_target_attack = 0
        attack_predictions = [0 for i in range(43)]

        # Extracts the information about the sticker amount and sticker coverage from the file name
        attack_properties = sticker_overlay_file.split('_')
        sticker_amount = attack_properties[2]
        sticker_coverage = attack_properties[-1].strip('%.png')

        # Loads the sticker overlay image
        sticker_overlay = Image.open(f"{attack_path}/sticker_overlays/{sticker_overlay_file}").convert('RGBA')
        sticker_overlay = toTensor(sticker_overlay).to(device)

        # Extracts the non transparent area of the sticker overlay
        overlay_mask = sticker_overlay[3] != 0

        # Defines the loading bar for iterating over all image samples from the dataset
        loading_bar_desc = f"{model_name}-{attack_algorithm} [Stickers:{sticker_amount}, " \
                           f"Class {victim_class} samples:{sample_counter}] Overall {dataset_type} samples"
        sample_iterator = trange(len(dataset[0]), desc=loading_bar_desc, leave=True, file=sys.stdout)

        for i in sample_iterator:
            # Skip samples that do not belong to the victim class
            if dataset[1][i] != int(victim_class):
                continue

            model.eval()
            clean_img = toTensor(dataset[0][i]).to(device)  # Loads the sample and converts it to torch
            prep_clean_img = utils.preprocess_images([clean_img], device)  # Preprocesses the sample image
            prediction_clean = model(prep_clean_img)  # Infers the clean image into the model

            # Calculates the prediction loss
            loss_clean += criterion(prediction_clean, target_output).item()
            # Checks whether the prediction was correct
            if torch.argmax(prediction_clean, 1) == dataset[1][i]:
                corrects_clean += 1
                clean_acc = corrects_clean / (sample_counter + 1) * 100  # Calculates the current accuracy

            # Extracts the image properties from the sample image and applies the sticker overlay
            activation = {}  # Resets the activation placeholder
            img_properties = utils.get_image_properties([clean_img], device)
            attacked_img = utils.sticker2images(prep_clean_img, img_properties, sticker_overlay, overlay_mask, device)
            prediction_attacked = model(attacked_img)  # Infers the attacked image into the model
            attacked_activation = activation  # Stores the sample feature maps

            # Calculates the prediction loss
            loss_attacked += criterion(prediction_attacked, target_output).item()
            # Checks whether the prediction was correct
            if torch.argmax(prediction_attacked, 1) == dataset[1][i]:
                corrects_attacked += 1
                attacked_acc = corrects_attacked / (sample_counter + 1) * 100  # Calculates the current accuracy

            # Calculates the target attack accuracy and loss
            if attack_target_class:
                loss_target_attack += criterion(prediction_attacked, attack_target_output).item()
                if torch.argmax(prediction_attacked, 1) == int(attack_target_class):
                    corrects_target_attack += 1
                    target_attack_acc = corrects_target_attack / (sample_counter + 1) * 100

            # Keeps track which classes have been predicted
            attack_predictions[torch.argmax(prediction_attacked, 1)] += 1

            if extract_dataframe:
                # Applies Monte Carlo Dropout to get confidence scores and variance values for each class
                mc_dropout_attacked_output, mc_dropout_attacked_variance = \
                    apply_mc_dropout(model, attacked_img, mc_dropout_num_repeats)

                if torch.argmax(mc_dropout_attacked_output) == dataset[1][i]:
                    correct_attacked_mc_dropout_preds += 1  # Checks if the Monte Carlo Dropout prediction was correct

                # Stores the prediction properties inside the pandas dataframe
                df.loc[sample_counter] = pandas.Series({
                    'sample_id': i,
                    'label': int(dataset[1][i]),
                    'img_attacked': np.moveaxis(utils.denormalize_image(attacked_img.squeeze(0).cpu()).numpy(), 0, -1),
                    'pred_attacked': int(torch.argmax(prediction_attacked.cpu(), 1)),
                    'conf_attacked': float(np.max(softmax(prediction_attacked).cpu().detach().numpy(), 1)),
                    'cscore_attacked': softmax(prediction_attacked).squeeze(0).cpu().detach().numpy(),
                    'layer_act_attacked': attacked_activation[target_layer].squeeze().cpu().detach().numpy(),
                    'cscore_mc_dropout_attacked': mc_dropout_attacked_output.cpu().numpy(),
                    'cvariance_mc_dropout_attacked': mc_dropout_attacked_variance.cpu().numpy()
                })

            # Updates the loading bar, based on the current prediction performance results
            loading_bar_desc = f"{model_name}-{attack_algorithm} [Stickers:{sticker_amount}, " \
                               f"Class {victim_class} samples:{sample_counter + 1}] Overall {dataset_type} samples"
            sample_iterator.set_description_str(loading_bar_desc)
            sample_iterator.set_postfix_str(f"Clean acc:{clean_acc:.1f}%, "
                                            f"Clean loss:{loss_clean / (sample_counter + 1):.4f}, "
                                            f"Attacked acc:{attacked_acc:.1f}%, "
                                            f"Attacked loss:{loss_attacked / (sample_counter + 1):.4f}")

            sample_counter += 1  # Increases the sample counter

        # Logs the attack and evaluation parameters to the console
        print(f"\nEvaluation results:")
        print(f"Victim Model:           {victim_model}")
        print(f"Victim Class:           {victim_class}")
        print(f"Attack Target Class:    {attack_target_class}")
        print(f"Attack Algorithm:       {attack_algorithm}")
        print(f"Sticker Amount:         {sticker_amount}")
        print(f"Sticker Coverage:       {sticker_coverage}%")
        print(f"Evaluation Model:       {model_name}")
        print(f"Evaluation Dataset:     {dataset_type}")
        print(f"Amount of Samples:      {sample_counter}")

        # Logs the final evaluation results to the console
        if attack_target_class:
            print(f"\nAccuracies:")
            print(f"-> Baseline class {victim_class} accuracy:              {clean_acc:.2f}%")
            print(f"-> Attacked class {victim_class} accuracy:              {attacked_acc:.2f}%")
            print(f"-> Target Attack class {attack_target_class} accuracy:         {target_attack_acc:.2f}%")
            print(f"Losses:")
            print(f"-> Baseline class {victim_class} loss:                  {loss_clean / sample_counter:.7f}")
            print(f"-> Attacked class {victim_class} loss:                  {loss_attacked / sample_counter:.7f}")
            print(
                f"-> Target Attack class {attack_target_class} loss:             {loss_target_attack / sample_counter:.7f}\n")
        else:
            print(f"\nAccuracies:")
            print(f"-> Baseline class {victim_class} accuracy:    {clean_acc:.2f}%")
            print(f"-> Attacked class {victim_class} accuracy:    {attacked_acc:.2f}%")
            print(f"Losses:")
            print(f"-> Baseline class {victim_class} loss:        {loss_clean / sample_counter:.7f}")
            print(f"-> Attacked class {victim_class} loss:        {loss_attacked / sample_counter:.7f}\n")

        if extract_dataframe:
            # Calculates the prediction accuracy when using Monte Carlo Dropout
            final_mc_dropout_acc = correct_attacked_mc_dropout_preds / sample_counter
            print('Model Performance using Monte Carlo Dropout:')
            print(f'- Accuracy: {final_mc_dropout_acc * 100:.2f}%\n')

            df_path = attack_path + '/evaluation_dataframes'  # Path for storing the evaluation dataframe
            if not os.path.exists(df_path): os.mkdir(df_path)  # Generates the evaluation dataframe path
            # Stores the dataframe as a pickle file inside the evaluation dataframe path
            df.to_pickle(f"{df_path}/{model_name}_GTSRB_{dataset_type}_set_activations_layer_{target_layer}_"
                         f"victim_class_{victim_class}_sticker_amount_{sticker_amount}.p")

        # Stores all attack information and evaluation results into a list
        csv_data = [attack_path, sticker_overlay_file, victim_model, victim_class, attack_target_class, attack_algorithm,
                    sticker_amount, sticker_coverage, model_name, dataset_type, sample_counter, clean_acc,
                    loss_clean / sample_counter, attacked_acc, loss_attacked / sample_counter, target_attack_acc,
                    loss_target_attack / sample_counter]
        csv_data.extend(attack_predictions)

        # Adds the list with the attack information and evaluation results to a csv file
        with open(csv_path, mode='a') as eval_csv:
            csv_writer = csv.writer(eval_csv, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(csv_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/evaluate_attack_config.yaml',
                        help='Path to the config yaml file, which contains the attack evaluation parameters.')
    parser.add_argument('--extract_dataframe', type=bool, default=True,
                        help='Whether the evaluation should be used for extracting a pandas dataframe, which will '
                             'contain different prediction properties for each evaluation sample.')
    parser.add_argument('--target_layer', type=str, default='layer4',
                        help='Name of the model layer (attribute from the model class), from which to extract the model'
                             'activations. Only used if extract_dataframe is set to True.')
    parser.add_argument('--mc_dropout_num_repeats', type=int, default=20,
                        help='Amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only used if '
                             'extract_dataframe is set to True.')
    parser.add_argument('--file_name', type=str, default='',
                        help='Specifies the name of a single sticker overlay file, which should be evaluated. If not '
                             'specified the evaluation will evaluate every sticker overlay file inside the '
                             'sticker_overlays folder.')
    args = parser.parse_args()

    # Reads the yaml file, which contains the evaluation parameters
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print(f"Initiating Realistic Sticker Attack evaluation for {cfg['model_name']} on the GTSRB {cfg['dataset']} set.")
    if args.extract_dataframe:
        print(f"Generating evaluation dataframe. Extracting model activations from layer {args.target_layer}. Monte "
              f"Carlo Dropout number of repeats set to {args.mc_dropout_num_repeats}.")

    # Displays all parameters for the current evaluation
    print('\nPARAMETERS:')
    for param in cfg.keys():
        print('    ' + param + ' = ' + str(cfg[param]))

    # Loads the classification model and the device
    model, device = utils.load_model(cfg['model_name'], cfg['device_ids'], cfg['weights_path'])

    # Loads the GTSRB dataset from the specified .pickle file
    print(f"Loading data from {cfg['dataset']}.p")
    with open(f"{cfg['data_path']}/{cfg['dataset']}.p", mode='rb') as f:
        data = pickle.load(f)
        samples = data['features']  # Image samples
        labels = data['labels']  # Labels
    print('Data loading finished! \n')

    criterion = nn.CrossEntropyLoss()  # Sets the loss function for evaluating the attack to cross entropy loss

    #  Evaluates the prediction performance on attacked samples
    evaluate_attack(
        model,
        cfg['model_name'],
        device,
        criterion,
        (samples, labels),
        cfg['dataset'],
        cfg['attack_path'],
        cfg['csv_path'],
        args.file_name,
        args.extract_dataframe,
        args.target_layer,
        args.mc_dropout_num_repeats
    )

    print('Evaluation finished.')


if __name__ == '__main__':
    main()
