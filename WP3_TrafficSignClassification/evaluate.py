import argparse
import glob
import os

from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch
import pandas
import yaml
from PIL import Image

from data import get_data_loader, denormalize_image
from train import load_model


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
            output = softmax(model(sample)[2])  # index 2 is for the actual output, 0 and 1 are pcl256 and pcl1024 outputs
            output_stack[i] = output  # Stacks the output scores

    # Returns the mean output score and variance value for each class
    return torch.mean(output_stack, dim=0), torch.var(output_stack, dim=0)
def eval_torchmetrics(labels, preds, config):
    # Convert lists to PyTorch tensors
    #print(preds)
    #print(labels)
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    # Initialize metrics
    accuracy = Accuracy(task='multiclass', num_classes=config['num_classes'], top_k=1)
    precision = Precision(task='multiclass',num_classes=config['num_classes'], average="macro", top_k=1)  # Assuming binary classification
    recall = Recall(task='multiclass',num_classes=config['num_classes'], average="macro", top_k=1)  # Assuming binary classification
    f1 = F1Score(task='multiclass',num_classes=config['num_classes'], average="macro", top_k=1)  # Assuming binary classification

    # Update metrics with predictions and ground truth
    #accuracy.update(preds, labels)
    #precision.update(preds, labels)
    #recall.update(preds, labels)
    #f1.update(preds, labels)

    # Calculate metric values
    accuracy_value = accuracy(preds, labels)*100
    precision_value = precision(preds, labels)*100
    recall_value = recall(preds, labels)*100
    f1_value = f1(preds, labels)*100

    print(f"Accuracy;  Precision; Recall; F1 Score")
    print(f"{accuracy_value:.2f} & {precision_value:.2f} & {recall_value:.2f} & {f1_value:.2f}")

def evaluate_model(model, criterion, data_loader, extract_dataframe, target_layer, mc_dropout_num_repeats, config):
    """
    Evaluates the model performance, using the given dataset. Stores different prediction properties for each image
    sample inside a pandas dataframe if extract_dataframe is set to True.

    :param model: The classification model, which is being evaluated.
    :param criterion: Loss function used for the evaluation.
    :param data_loader: The dataloader containing the given dataset.
    :param extract_dataframe: Whether the evaluation should be used for extracting a pandas dataframe, which will
    contain different prediction properties for each evaluation sample.
    :param target_layer: Name of the model layer (attribute from the model class), from which to extract the model
    activations. Only used if extract_dataframe is set to True.
    :param mc_dropout_num_repeats: Amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only used
    if extract_dataframe is set to True.
    :return: The prediction accuracy and prediction loss for the model on the given dataset. If extract_dataframe is
    set to True, returns also a pandas dataframe containing different prediction properties for each evaluation sample.
    """
    correct_preds = 0  # Tracks the amount of correct predictions
    preds_loss = 0  # Tracks the prediction loss
    total_samples = 0  # Tracks the total amount of samples used for the evaluation
    df = None  # Placeholder for the pandas dataframe

    if extract_dataframe:
        correct_mc_dropout_preds = 0  # Tracks the amount of correct predictions when using Monte Carlo Dropout
        activation = {}  # Used for storing the target layer activations (feature maps)

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = getattr(model.module, target_layer)  # Extracts the given layer attribute from the model class
        # Extracts the feature maps from the target layer and stores them in activation
        layer.register_forward_hook(get_activation(target_layer))

        # Defines the pandas dataframe, used for storing the model prediction properties for each evaluation sample
        df = pandas.DataFrame(index=range(len(data_loader)), columns=[
            'sample_id',
            'label',
            'img_clean',
            'pred_clean',
            'conf_clean',
            'cscore_clean',
            'layer_act_clean',
            'cscore_mc_dropout_clean',
            'cvariance_mc_dropout_clean'
        ])

        softmax = nn.Softmax(dim=1)  # Applies softmax to the final output scores
    preds=[]
    labels=[]


    with tqdm(data_loader, desc='Evaluation progress') as pbar:
        # Iterates over all dataset batches
        for _i, (x, y) in enumerate(pbar):
            #if _i>10: break
            # Iterates over all samples from the batch
            for i in range(x.shape[0]):
                model.eval()
                activation = {}  # Resets the activation placeholder
                img = x[i]  # Current image sample
                prox256, prox1024, output = model(img.unsqueeze(0))  # Infers the single image sample into the model
                clean_activation = activation  # Stores the sample feature maps

                preds_loss += criterion(output, y[i].unsqueeze(0)).item()  # Calculates the prediction loss
                if torch.argmax(output, 1) == y[i]:
                    correct_preds += 1  # Checks if the prediction was correct

                if extract_dataframe:
                    # Applies Monte Carlo Dropout to get confidence scores and variance values for each class
                    mc_dropout_output, mc_dropout_variance = \
                        apply_mc_dropout(model, img.unsqueeze(0), mc_dropout_num_repeats)

                    if torch.argmax(mc_dropout_output) == y[i].to('cpu'):
                        correct_mc_dropout_preds += 1  # Checks if the Monte Carlo Dropout prediction was correct

                    # Stores the prediction properties inside the pandas dataframe
                    df.loc[total_samples] = pandas.Series({
                        'sample_id': total_samples,
                        'label': int(y[i].cpu()),
                        'img_clean': np.moveaxis(denormalize_image(x[i].cpu()).numpy(), 0, -1),
                        'pred_clean': int(torch.argmax(output.cpu(), 1)),
                        'conf_clean': float(np.max(softmax(output).cpu().detach().numpy(), 1)),
                        'cscore_clean': softmax(output).squeeze(0).cpu().detach().numpy(),
                        'layer_act_clean': clean_activation[target_layer].squeeze().cpu().detach().numpy(),
                        'cscore_mc_dropout_clean': mc_dropout_output.cpu().numpy(),
                        'cvariance_mc_dropout_clean': mc_dropout_variance.cpu().numpy()
                    })
                preds.append(int(y[i].cpu()))
                labels.append(int(torch.argmax(output.cpu(), 1)))
                total_samples += 1  # Increases the sample counter

    eval_torchmetrics(labels, preds, config)

    # Calculates the final accuracy and loss, after all batches have been loaded
    final_acc = correct_preds / total_samples  # Calculates the prediction accuracy for the dataset
    final_loss = preds_loss / total_samples  # Calculates the prediction loss for the dataset

    if extract_dataframe:
        # Calculates the prediction accuracy when using Monte Carlo Dropout
        final_mc_dropout_acc = correct_mc_dropout_preds / total_samples
        print('\nModel Performance using Monte Carlo Dropout:')
        print(f'- Accuracy: {final_mc_dropout_acc*100:.2f}%\n')

    # Logs the accuracy and loss for the dataset to the console
    print('Model Performance:')
    print(f'- Accuracy: {final_acc*100:.2f}%\t'
          f'Loss: {final_loss:.6f}\n')

    return final_acc*100, final_loss, df


def evaluate_model_single_images(model, criterion, folder_path,folder_label, extract_dataframe, target_layer, mc_dropout_num_repeats, device):
    """
    Evaluates the model performance, using the given data as images folder. Stores different prediction properties for each image
    sample inside a pandas dataframe if extract_dataframe is set to True.

    :param model: The classification model, which is being evaluated.
    :param criterion: Loss function used for the evaluation.
    :param folder_path: The path to the folder with images.
    :param folder_label: The label of all the images in the folder.
    :param extract_dataframe: Whether the evaluation should be used for extracting a pandas dataframe, which will
    contain different prediction properties for each evaluation sample.
    :param target_layer: Name of the model layer (attribute from the model class), from which to extract the model
    activations. Only used if extract_dataframe is set to True.
    :param mc_dropout_num_repeats: Amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only used
    if extract_dataframe is set to True.
    :param device: The gpu index
    :return: The prediction accuracy and prediction loss for the model on the given dataset. If extract_dataframe is
    set to True, returns also a pandas dataframe containing different prediction properties for each evaluation sample.
    """
    correct_preds = 0  # Tracks the amount of correct predictions
    preds_loss = 0  # Tracks the prediction loss
    total_samples = 0  # Tracks the total amount of samples used for the evaluation
    df = None  # Placeholder for the pandas dataframe

    if extract_dataframe:
        correct_mc_dropout_preds = 0  # Tracks the amount of correct predictions when using Monte Carlo Dropout
        activation = {}  # Used for storing the target layer activations (feature maps)

        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        layer = getattr(model.module, target_layer)  # Extracts the given layer attribute from the model class
        # Extracts the feature maps from the target layer and stores them in activation
        layer.register_forward_hook(get_activation(target_layer))

        # Defines the pandas dataframe, used for storing the model prediction properties for each evaluation sample
        df = pandas.DataFrame(index=range(len(os.listdir(folder_path))), columns=[
            'sample_id',
            'label',
            'img_clean',
            'pred_clean',
            'conf_clean',
            'cscore_clean',
            'layer_act_clean',
            'cscore_mc_dropout_clean',
            'cvariance_mc_dropout_clean'
        ])

        softmax = nn.Softmax(dim=1)  # Applies softmax to the final output scores
    folder_label.to(device)
    for img_path in sorted(glob.glob(folder_path+'/*.png')):


        model.eval()
        activation = {}  # Resets the activation placeholder
        img = Image.open(img_path)  # Current image sample
        img.to(device)
        output = model(img.unsqueeze(0))  # Infers the single image sample into the model
        clean_activation = activation  # Stores the sample feature maps

        preds_loss += criterion(output, folder_label).item()  # Calculates the prediction loss
        if torch.argmax(output, 1) == folder_path:
            correct_preds += 1  # Checks if the prediction was correct

        if extract_dataframe:
            # Applies Monte Carlo Dropout to get confidence scores and variance values for each class
            mc_dropout_output, mc_dropout_variance = \
                apply_mc_dropout(model, img.unsqueeze(0), mc_dropout_num_repeats)

            if torch.argmax(mc_dropout_output) == folder_path.to('cpu'):
                correct_mc_dropout_preds += 1  # Checks if the Monte Carlo Dropout prediction was correct

            # Stores the prediction properties inside the pandas dataframe
            df.loc[total_samples] = pandas.Series({
                'sample_id': total_samples,
                'label': int(y[i].cpu()),
                'img_clean': np.moveaxis(denormalize_image(img.cpu()).numpy(), 0, -1),
                'pred_clean': int(torch.argmax(output.cpu(), 1)),
                'conf_clean': float(np.max(softmax(output).cpu().detach().numpy(), 1)),
                'cscore_clean': softmax(output).squeeze(0).cpu().detach().numpy(),
                'layer_act_clean': clean_activation[target_layer].squeeze().cpu().detach().numpy(),
                'cscore_mc_dropout_clean': mc_dropout_output.cpu().numpy(),
                'cvariance_mc_dropout_clean': mc_dropout_variance.cpu().numpy()
            })

        total_samples += 1  # Increases the sample counter

    # Calculates the final accuracy and loss, after all batches have been loaded
    final_acc = correct_preds / total_samples  # Calculates the prediction accuracy for the dataset
    final_loss = preds_loss / total_samples  # Calculates the prediction loss for the dataset

    if extract_dataframe:
        # Calculates the prediction accuracy when using Monte Carlo Dropout
        final_mc_dropout_acc = correct_mc_dropout_preds / total_samples
        print('\nModel Performance using Monte Carlo Dropout:')
        print(f'- Accuracy: {final_mc_dropout_acc*100:.2f}%\n')

    # Logs the accuracy and loss for the dataset to the console
    print('Model Performance:')
    print(f'- Accuracy: {final_acc*100:.2f}%\t'
          f'Loss: {final_loss:.6f}\n')

    return final_acc*100, final_loss, df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the evaluation parameters.')
    parser.add_argument('--extract_dataframe', type=bool, default=False,
                        help='Whether the evaluation should be used for extracting a pandas dataframe, which will '
                             'contain different prediction properties for each evaluation sample.')
    parser.add_argument('--target_layer', type=str, default='avgpool',
                        help='Name of the model layer (attribute from the model class), from which to extract the model'
                             'activations. Only used if extract_dataframe is set to True.')
    parser.add_argument('--mc_dropout_num_repeats', type=int, default=2,
                        help='Amount of repeats for every sample to be inferred for Monte Carlo Dropout. Only used if '
                             'extract_dataframe is set to True.')
    args = parser.parse_args()

    # Reads the yaml file, which contains the evaluation parameters
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print(f"Initiating GTSRB {cfg['dataset']} set evaluation for the {cfg['model_name']} model.")
    if args.extract_dataframe:
        print(f"Generating evaluation dataframe. Extracting model activations from layer {args.target_layer}. Monte "
              f"Carlo Dropout number of repeats set to {args.mc_dropout_num_repeats}.")

    # Displays all parameters for the current evaluation
    print('\nPARAMETERS')
    for param in cfg.keys():
        print('    ' + param + ' = ' + str(cfg[param]))

    # Loads the classification model and the device
    model, device = load_model(cfg['model_name'], cfg['device_ids'], cfg, f"{cfg['weights_path']}")

    # Loads the specified dataset and dataloader for the evaluation
    data_loader = get_data_loader(cfg['data_path'], cfg['dataset'], cfg['batch_size'], device)

    criterion = nn.CrossEntropyLoss()  # Sets the loss function for evaluating the model to cross entropy loss

    # Initiates the evaluation process
    print('Initiating the evaluation.')
    acc, loss, df = \
        evaluate_model(
            model,
            criterion,
            data_loader,
            args.extract_dataframe,
            args.target_layer,
            args.mc_dropout_num_repeats,
            cfg
        )

    # Extracts the session path from the model weights path
    session_path = '/'.join(cfg['weights_path'].split('/')[:-1])

    if args.extract_dataframe:
        df_path = session_path + '/evaluation_dataframes'  # Path for storing the evaluation dataframe
        if not os.path.exists(df_path): os.mkdir(df_path)  # Generates the evaluation dataframe path
        # Stores the dataframe as a pickle file inside the evaluation dataframe path
        df.to_pickle(f"{df_path}/{cfg['model_name']}_GTSRB_{cfg['dataset']}_set_activations_layer_{args.target_layer}.p")

    # Instantiates a TensorBoard SummaryWriter for logging the evaluation results to TensorBoard
    if not os.path.exists(f"{session_path}/TensorBoard_Logs/eval_{cfg['dataset']}"):
        writer = SummaryWriter(f"{session_path}/TensorBoard_Logs/eval_{cfg['dataset']}")
        writer.add_scalar(f"Final_Evaluation_Results/{cfg['dataset']}_Accuracy", acc)
        writer.add_scalar(f"Final_Evaluation_Results/{cfg['dataset']}_Loss", loss)
        writer.close()

    print('Evaluation finished.')


if __name__ == '__main__':
    main()
