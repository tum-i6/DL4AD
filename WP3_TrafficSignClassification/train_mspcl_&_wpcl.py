"Training only in MSPCL mode!"


import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
import argparse
import shutil
import os

from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import pcl_resnet_models
import pcl_vgg_models
import pcl_inception_models
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import yaml
from pcl.proximity import Proximity
from pcl.contrastive_proximity import Con_Proximity
from data import get_train_loaders
from functools import reduce
PCL_custom_labels_gtsrb={
    0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0,
    9:1, 10:1, 15:1, 16:1,
    11:2, 18:2, 19:2, 20:2, 21:2, 22:2, 23:2, 24:2, 25:2, 26:2, 27:2, 28:2, 29:2, 30:2, 31:2,
    32:3, 41:3, 42:3,
    33:4, 34:4, 35:4, 36:4, 37:4, 38:4, 39:4, 40:4,
    12:5, 13:6, 14:7, 17:8
}

PCL_custom_labels_imagenet={
    0:0, 1:0, 2:0,
    3:1, 4:1, 5:1, 6:1,
    7:2, 8:2, 9:2, 10:2,
    11:3, 12:3, 13:3, 14:3
}

PCL_custom_labels_cure_tsd={
    1:0, 2:0,
    3:1, 4:1,
    8:2, 9:2, 10:2,
    0:3, 5:4, 6:5, 7:6, 11:7, 12:8, 13:9
}


# Model zoo (all supported models)
models = {'VGG16': pcl_vgg_models.vgg16_bn(pretrained=True),
          'VGG19': pcl_vgg_models.vgg19_bn(pretrained=True),
          'InceptionV3': pcl_inception_models.inception_v3(pretrained=True),
          'ResNet18': pcl_resnet_models.resnet18(pretrained=True),
          'ResNet50': pcl_resnet_models.resnet50(pretrained=True),
          'ResNet101': pcl_resnet_models.resnet101(pretrained=True),
          'ResNet152': pcl_resnet_models.resnet152(pretrained=True),
          'WideResNet50': pcl_resnet_models.wide_resnet50_2(pretrained=True),
          'WideResNet101': pcl_resnet_models.wide_resnet101_2(pretrained=True),
          'ResNeXt50': pcl_resnet_models.resnext50_32x4d(pretrained=True),
          'ResNeXt101': pcl_resnet_models.resnext101_32x8d(pretrained=True)}
def PCL_step_custom_prox(custom_clusters, config):

    for cluster in custom_clusters:
        prox=custom_clusters[cluster]['prox']

        #for param in prox['criterion_prox_1'].parameters():
        #    param.grad.data *= (1000. / config['pcl_weight-prox'])
        #    if not torch.isfinite(param.grad).all():
        #        print('got infinite! on prox 1')
        prox['optimizer_prox_1'].step()

        #for  param in prox['criterion_prox_2'].parameters():
        #    param.grad.data *= (1000. / config['pcl_weight-prox'])
        #    if not torch.isfinite(param.grad).all():
        #        print('got infinite! on prox 2')
        prox['optimizer_prox_2'].step()



        #for param in prox['criterion_conprox_1'].parameters():
        #    param.grad.data *= (1000. / config['pcl_weight-conprox'])
        prox['optimizer_conprox_1'].step()

        #for param in prox['criterion_conprox_2'].parameters():
        #    param.grad.data *= (1000. / config['pcl_weight-conprox'])
        prox['optimizer_conprox_2'].step()



def PCL_custom_loss(custom_clusters, feats1,feats2, y, config, prox_centers_1, prox_centers_2, conprox_center_1, conprox_center_2):
    y_np = y.cpu().numpy()
    loss_prox_1=0
    loss_prox_2=0
    loss_conprox_1=0
    loss_conprox_2=0
    misses=0
    for i, cluster in enumerate(custom_clusters):
        masks=[y_np==x for x in custom_clusters[cluster]['classes']]
        total_mask = torch.from_numpy(reduce(np.logical_or, masks)).cuda()
        _y=torch.masked_select(y, total_mask)
        if len(_y)==0:
            misses+=1
            continue
        _feats1=feats1[total_mask]
        _feats2=feats2[total_mask]
        lp1=custom_clusters[cluster]['prox']['criterion_prox_1'](_feats1, _y, has_custom_classes=True,
                    custom_classes=np.asarray(custom_clusters[cluster]['classes']), custom_prox_center=prox_centers_1[i])


        lp2=custom_clusters[cluster]['prox']['criterion_prox_2'](_feats2, _y,has_custom_classes=True,
                    custom_classes=np.asarray(custom_clusters[cluster]['classes']), custom_prox_center=prox_centers_2[i])
        loss_prox_1 += lp1
        loss_prox_2 += lp2

        lcp1=custom_clusters[cluster]['prox']['criterion_conprox_1'](_feats1, _y,has_custom_classes=True,
                    custom_classes=np.asarray(custom_clusters[cluster]['classes']),custom_conprox_center=conprox_center_1[i])
        lcp2=custom_clusters[cluster]['prox']['criterion_conprox_2'](_feats2, _y,has_custom_classes=True,
                    custom_classes=np.asarray(custom_clusters[cluster]['classes']), custom_conprox_center=conprox_center_2[i])

        if torch.isnan(lp1) or torch.isnan(lp2) or torch.isnan(lcp1) or torch.isnan(lcp2):
            print('get stuck')
        loss_conprox_1 += lcp1
        loss_conprox_2 += lcp2

    loss_prox_1/=(5-misses)
    loss_prox_2/=(5-misses)
    loss_conprox_1/=(5-misses)
    loss_conprox_2/=(5-misses)
    loss_conprox_1 *= config['pcl_weight-conprox']
    loss_conprox_2 *= config['pcl_weight-conprox']
    return loss_prox_1, loss_prox_2, loss_conprox_1, loss_conprox_2


def train_batch(model, model_name, loss_func, x, y, opt, mspcl_optimizer_prox_2,
                mspcl_optimizer_prox_1, mspcl_optimizer_conprox_2, mspcl_optimizer_conprox_1, mspcl_criterion_prox_2,
                mspcl_criterion_prox_1, mspcl_criterion_conprox_2, mspcl_criterion_conprox_1, mspcl_custom_clusters,
                wpcl_optimizer_prox_256, wpcl_optimizer_prox_1024, wpcl_optimizer_conprox_256, wpcl_optimizer_conprox_1024,
                wpcl_criterion_prox_2, wpcl_criterion_prox_1, wpcl_criterion_conprox_256, wpcl_criterion_conprox_1024,
                config):
    """
    Infers a single training batch into the model and calculates the corresponding loss and accuracy. Updates the model
    weights based on the prediction loss.

    :param model: The classification model, which is being trained.
    :param model_name: Name of the classification model, which is being trained. Mandatory since InceptionV3 contains 2
    outputs and 2 losses.
    :param loss_func: Loss function used for optimization.
    :param x: Batch of training samples (images).
    :param y: Batch of labels for the training samples.
    :param opt: Optimizer used for training the model.
    :return: The prediction accuracy, amount of correct predictions, average loss value and the amount of training
    samples for the current batch.
    """
    opt.zero_grad()

    # InceptionV3, returns 2 outputs during training
    if model_name == 'InceptionV3':
        feats2, feats1, output, aux_output = model(x)  # Infers the training batch into the model
        loss1 = loss_func(output, y)  # Calculates the final output layer loss
        loss2 = loss_func(aux_output, y)  # Calculates the auxiliary output layer loss
        loss = loss1 + 0.4*loss2  # Calculates the final loss for InceptionV3
    else:
        feats2, feats1, output = model(x)  # Infers the training batch into the model
        loss = loss_func(output, y)  # Calculates the loss

    y_custom=PCL_get_custom_label(y, config['dataset'])
    mspcl_loss_prox_1 = mspcl_criterion_prox_1(feats1, y_custom)
    mspcl_loss_prox_2 = mspcl_criterion_prox_2(feats2, y_custom)

    mspcl_loss_conprox_1 = mspcl_criterion_conprox_1(feats1, y_custom)
    mspcl_loss_conprox_2 = mspcl_criterion_conprox_2(feats2, y_custom)
    # MSPCL
    mspcl_custom_loss_prox_1, mspcl_custom_loss_prox_2, mspcl_custom_loss_conprox_1, mspcl_custom_loss_conprox_2=PCL_custom_loss(mspcl_custom_clusters, feats1, feats2, y, config,
                                                                                                         mspcl_criterion_prox_1.centers, mspcl_criterion_prox_2.centers,
                                                                                                         mspcl_criterion_conprox_1.centers, mspcl_criterion_conprox_2.centers)

    #---------old version- in the cvpr paper
    #mspcl_loss_prox_1=(mspcl_loss_prox_1+custom_loss_prox_1)/2
    #mspcl_loss_prox_2=(mspcl_loss_prox_2+custom_loss_prox_2)/2

    #mspcl_loss_conprox_1 = (mspcl_loss_conprox_1 + mspcl_custom_loss_conprox_1) / 2
    #mspcl_loss_conprox_2 = (mspcl_loss_conprox_2 + mspcl_custom_loss_conprox_2) / 2

    #--------new version MSPCL
    mspcl_loss_prox_1 = mspcl_loss_prox_1 + mspcl_custom_loss_prox_1
    mspcl_loss_prox_2 = mspcl_loss_prox_2 + mspcl_custom_loss_prox_2

    mspcl_loss_conprox_1 = mspcl_loss_conprox_1 + mspcl_custom_loss_conprox_1
    mspcl_loss_conprox_2 = mspcl_loss_conprox_2 + mspcl_custom_loss_conprox_2

    mspcl_loss_prox_1 *= config['pcl_weight-prox']
    mspcl_loss_prox_2 *= config['pcl_weight-prox']

    mspcl_loss_conprox_1 *= config['pcl_weight-conprox']
    mspcl_loss_conprox_2 *= config['pcl_weight-conprox']

    # WPCL
    wpcl_loss_prox_1 = wpcl_criterion_prox_1(feats1, y)
    wpcl_loss_prox_2 = wpcl_criterion_prox_2(feats2, y)

    wpcl_loss_conprox_1 = wpcl_criterion_conprox_1024(feats1, y, custom_distance=config['pcl_custom_distance'],
                                               custom_dist_rate=config['pcl_custom_distance_rate'])
    wpcl_loss_conprox_2 = wpcl_criterion_conprox_256(feats2, y, custom_distance=config['pcl_custom_distance'],
                                             custom_dist_rate=config['pcl_custom_distance_rate'])

    wpcl_loss_prox_1 *= config['pcl_weight-prox']
    wpcl_loss_prox_2 *= config['pcl_weight-prox']

    wpcl_loss_conprox_1 *= config['pcl_weight-conprox']
    wpcl_loss_conprox_2 *= config['pcl_weight-conprox']

    loss_prox_1= 0.5*mspcl_loss_prox_1+0.5*wpcl_loss_prox_1
    loss_prox_2= 0.5*mspcl_loss_prox_2+0.5*wpcl_loss_prox_2


    loss_conprox_1= 0.5*mspcl_loss_conprox_1+0.5*wpcl_loss_conprox_1
    loss_conprox_2= 0.5*mspcl_loss_conprox_2+0.5*wpcl_loss_conprox_2




    loss=loss + loss_prox_1 + loss_prox_2  - loss_conprox_1 - loss_conprox_2 # total loss
    mspcl_optimizer_prox_1.zero_grad()
    mspcl_optimizer_prox_2.zero_grad()

    mspcl_optimizer_conprox_1.zero_grad()
    mspcl_optimizer_conprox_2.zero_grad()
    loss.backward()
    opt.step()

    for param in mspcl_criterion_prox_1.parameters():
        param.grad.data *= (1. / config['pcl_weight-prox'])
    mspcl_optimizer_prox_1.step()

    for param in mspcl_criterion_prox_2.parameters():
        param.grad.data *= (1. / config['pcl_weight-prox'])
    mspcl_optimizer_prox_2.step()

    for param in mspcl_criterion_conprox_1.parameters():
        param.grad.data *= (1. / config['pcl_weight-conprox'])
    mspcl_optimizer_conprox_1.step()

    for param in mspcl_criterion_conprox_2.parameters():
        param.grad.data *= (1. / config['pcl_weight-conprox'])
    mspcl_optimizer_conprox_2.step()

    PCL_step_custom_prox(mspcl_custom_clusters, config)
    with torch.no_grad():
        corrects = torch.sum(torch.argmax(output, 1) == y)  # Calculates the amount of correct predictions
        acc = (corrects / len(x)) * 100  # Calculates the batch accuracy based on the amount of correct predictions

    return acc.item(), corrects.item(), loss.item(), len(x)


def valid_batch(model,model_name, loss_func, x, y, criterion_prox_256, criterion_prox_1024, criterion_conprox_256,
                criterion_conprox_1024):
    """
    Infers a single (validation) batch into the model and calculates the corresponding loss and accuracy.

    :param model: The classification model, which is being evaluated.
    :param loss_func: Loss function used for tracking the loss on the validation data.
    :param x: Batch of validation samples (images).
    :param y: Batch of labels for the validation samples.
    :return: The prediction accuracy, amount of correct predictions, average loss value and the amount of validation
    samples for the current batch.
    """
    if model_name == 'InceptionV3':
        feats256, feats1024, output = model(x)
    else:
        feats256, feats1024, output = model(x)  # Infers the validation batch into the model
    loss = loss_func(output, y)  # Calculates the loss

    # PCL
    #loss_prox_1024 = criterion_prox_1024(feats1024, y)
    #loss_prox_256 = criterion_prox_256(feats256, y)

    #loss_conprox_1024 = criterion_conprox_1024(feats1024, y)
    #loss_conprox_256 = criterion_conprox_256(feats256, y)

    #loss = loss + loss_prox_1024 + loss_prox_256 - loss_conprox_1024 - loss_conprox_256  # total loss

    corrects = torch.sum(torch.argmax(output, 1) == y)  # Calculates the amount of correct predictions
    acc = (corrects / len(x)) * 100  # Calculates the batch accuracy based on the correct predictions

    return acc.item(), corrects.item(), loss.item(), len(x)


def train_epoch(model, model_name, train_loader, epoch, num_epochs, criterion, opt, mspcl_optimizer_prox_2,
                mspcl_optimizer_prox_1, mspcl_optimizer_conprox_2, mspcl_optimizer_conprox_1, mspcl_criterion_prox_2,
                mspcl_criterion_prox_1, mspcl_criterion_conprox_2, mspcl_criterion_conprox_1, mspcl_custom_clusters,
                wpcl_optimizer_prox_256, wpcl_optimizer_prox_1024, wpcl_optimizer_conprox_256, wpcl_optimizer_conprox_1024,
                wpcl_criterion_prox_2, wpcl_criterion_prox_1, wpcl_criterion_conprox_256, wpcl_criterion_conprox_1024, config):
    """
    Trains the model for a single epoch using the train dataset and optimizes its weights.

    :param model: The classification model, which is being trained.
    :param model_name: Name of the classification model, which is being trained.
    :param train_loader: The dataloader containing the training batches.
    :param epoch: The integer value of the current epoch.
    :param num_epochs: Amount of training epochs.
    :param criterion: Loss function used for optimization.
    :param opt: Optimizer used for training the model.
    :return: The average accuracy and loss during training.
    """
    model.train()
    epoch_corrects = 0  # Used to track the amount of correct predictions for the current epoch during training
    epoch_loss = 0  # Used to track the loss for the current epoch during training
    total_nums = 0  # Used to track the total amount of samples used for training the current epoch

    with tqdm(train_loader, desc=f'{model_name} [Epoch:{epoch + 1}/{num_epochs}]') as pbar:
        # Iterates over all training batches
        for x, y in pbar:
            # Infers the batch into the model and calculates the accuracy, amount of correct predictions, loss and
            # amount of samples used inside the training batch
            acc, corrects, loss, nums = train_batch(model, model_name, criterion, x, y, opt, mspcl_optimizer_prox_2,
                                                    mspcl_optimizer_prox_1, mspcl_optimizer_conprox_2, mspcl_optimizer_conprox_1, mspcl_criterion_prox_2,
                                                    mspcl_criterion_prox_1, mspcl_criterion_conprox_2, mspcl_criterion_conprox_1, mspcl_custom_clusters,
                                                    wpcl_optimizer_prox_256, wpcl_optimizer_prox_1024, wpcl_optimizer_conprox_256, wpcl_optimizer_conprox_1024,
                                                    wpcl_criterion_prox_2, wpcl_criterion_prox_1, wpcl_criterion_conprox_256, wpcl_criterion_conprox_1024, config)
            with torch.no_grad():
                # Logs the train performance for the current batch to the console
                pbar.set_postfix(
                    {'Batch Loss': round(loss, 4), 'Batch Accuracy(%)': round(acc, 2), 'Batch size': nums}
                )
                epoch_corrects += corrects  # Tracks the amount of correct predictions for the current epoch
                epoch_loss += loss * nums  # Tracks the loss for the current epoch
                total_nums += nums  # Tracks the amount of samples used for training the current epoch

    # Calculates the final train accuracy and loss, after all training batches have been loaded
    train_acc = epoch_corrects / total_nums  # Calculates the average accuracy for the current epoch
    train_loss = epoch_loss / total_nums  # Calculates the average loss for the current epoch

    return train_acc, train_loss


def validate_epoch(model,model_name, val_loader, criterion, criterion_prox_256,
            criterion_prox_1024, criterion_conprox_256, criterion_conprox_1024):
    """
    Evaluates the model using the validation dataset.

    :param model: The classification model, which is being evaluated.
    :param val_loader: The dataloader containing the validation batches.
    :param criterion: Loss function used for tracking the loss on the validation data.
    :return: The average accuracy and loss for the validation dataset.
    """
    model.eval()
    epoch_corrects = 0  # Used to track the amount of correct predictions for the current epoch during validation
    epoch_loss = 0  # Used to track the loss for the current epoch during validation
    total_nums = 0  # Used to track the amount of samples used for validating the current epoch

    with torch.no_grad():
        # Iterates over all validation batches
        for x, y in val_loader:
            # Infers the batch into the model and calculates the accuracy, amount of correct predictions, loss and
            # amount of samples used for the validation batch
            acc, corrects, loss, nums = valid_batch(model,model_name, criterion, x, y, criterion_prox_256,
            criterion_prox_1024, criterion_conprox_256, criterion_conprox_1024)
            epoch_corrects += corrects  # Tracks the amount of correct predictions for the current epoch
            epoch_loss += loss * nums  # Tracks the loss for the current epoch
            total_nums += nums  # Tracks the amount of samples used for validating the current epoch

        # Calculates the final validation accuracy and loss, after all validation batches have been loaded
        val_acc = epoch_corrects / total_nums  # Calculates the average accuracy for the current epoch
        val_loss = epoch_loss / total_nums  # Calculates the average loss for the current epoch

    return val_acc, val_loss


def train_model(num_epochs, patience, model, model_name, criterion, opt, train_loader, val_loader, session_path,
                train_writer, val_writer, mspcl_optimizer_prox_2, mspcl_optimizer_prox_1, mspcl_optimizer_conprox_2,
                mspcl_optimizer_conprox_1, mspcl_criterion_prox_2, mspcl_criterion_prox_1, mspcl_criterion_conprox_2,
                mspcl_criterion_conprox_1, mspcl_custom_clusters, wpcl_optimizer_prox_256, wpcl_optimizer_prox_1024,
                wpcl_optimizer_conprox_256, wpcl_optimizer_conprox_1024, wpcl_criterion_prox_2, wpcl_criterion_prox_1,
                wpcl_criterion_conprox_256, wpcl_criterion_conprox_1024, config):
    """
    Trains the classification model using the train dataset and optimizes its weights. After each epoch, the model is
    evaluated using the validation dataset and if the validation loss decreases, the weights are stored.

    :param num_epochs: Amount of training epochs.
    :param patience: Amount of training epochs without any performance increase after which the training should end.
    :param model: The classification model, which is being trained.
    :param model_name: Name of the classification model, which is being trained.
    :param criterion: Loss function used for optimization.
    :param opt: Optimizer used for training the model.
    :param train_loader: The dataloader containing the training batches.
    :param val_loader: The dataloader containing the validation batches.
    :param session_path: Path to the output directory for storing the training results and TensorBoard logfiles.
    :param train_writer: TensorBoard instance used for logging the training results to TensorBoard.
    :param val_writer: TensorBoard instance used for logging the validation results to TensorBoard.
    :return: The train accuracy & loss and val accuracy & loss for the model weights, which achieved the minimum
    validation loss.
    """
    wait = 0  # Used to count the amount of epochs without any validation loss progress (used for early stopping)
    valid_loss_min = np.Inf  # Initial value for tracking the minimum validation loss value
    weights_name = ''  # Placeholder for the .pt file name of the best model weights (will contain accuracy value)

    # Executes the training for num_epochs
    for epoch in range(num_epochs):
        # Trains the model for one epoch and returns the average accuracy and loss during training
        train_acc, train_loss = train_epoch(model, model_name, train_loader, epoch, num_epochs, criterion, opt, mspcl_optimizer_prox_2,
                                            mspcl_optimizer_prox_1, mspcl_optimizer_conprox_2, mspcl_optimizer_conprox_1, mspcl_criterion_prox_2,
                                            mspcl_criterion_prox_1, mspcl_criterion_conprox_2, mspcl_criterion_conprox_1, mspcl_custom_clusters,
                                            wpcl_optimizer_prox_256, wpcl_optimizer_prox_1024, wpcl_optimizer_conprox_256, wpcl_optimizer_conprox_1024,
                                            wpcl_criterion_prox_2, wpcl_criterion_prox_1, wpcl_criterion_conprox_256, wpcl_criterion_conprox_1024, config)

        # Evaluates the model using the validation dataset and returns the average accuracy and loss
        val_acc, val_loss = validate_epoch(model, model_name, val_loader, criterion, mspcl_criterion_prox_2,
                                           mspcl_criterion_prox_1, mspcl_criterion_conprox_2, mspcl_criterion_conprox_1)

        # Logs the performance summary for the current epoch
        print(f"- Train loss: {train_loss:.6f}\t"
              f"Train accuracy: {train_acc*100:.2f}%\t\n"
              f"- Val loss:   {val_loss:.6f}\t"
              f"Val accuracy:   {val_acc*100:.2f}%\t")

        # Logs the train and validations loss & accuracy for the current epoch to TensorBoard
        train_writer.add_scalar('Training/Loss', train_loss, epoch)
        val_writer.add_scalar('Training/Loss', val_loss, epoch)
        train_writer.add_scalar('Training/Accuracy', train_acc*100, epoch)
        val_writer.add_scalar('Training/Accuracy', val_acc*100, epoch)
        train_writer.close()
        val_writer.close()

        # Save model weights if validation loss has decreased
        if val_loss <= valid_loss_min:
            print(f"- Validation loss decreased ({valid_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
            if weights_name != '':
                os.remove(f"{session_path}/{weights_name}")  # Removes the old model weights file
            weights_name = f"weights_{round(val_acc*100, 2)}%.pt"  # Model weights file contains the validation accuracy
            torch.save(model.state_dict(), f"{session_path}/{weights_name}")  # Stores the new model weights
            valid_loss_min = val_loss  # Overwrites the minimum validation loss value
            wait = 0  # Resets the patience counter

            # Stores the new (so far best) performance values
            best_train_acc = train_acc * 100
            best_train_loss = train_loss
            best_val_acc = val_acc * 100
            best_val_loss = val_loss

        # Early stopping
        else:
            wait += 1  # Tracks the patience
            if wait == patience:
                print(f"Terminated training for early stopping at epoch {epoch+1}.")
                return best_train_acc, best_train_loss, best_val_acc, best_val_loss
            else:
                print(f"Current patience counter: {wait}, {patience-wait} more to go!")
        print('')

    return best_train_acc, best_train_loss, best_val_acc, best_val_loss


def load_model(model_name, device_ids, config, weights_path=None):
    """
    Loads the classification model and the device.

    :param model_name: Name of the classification model, which should be loaded.
    :param device_ids: Device IDs for the GPUs, which should be used for training/evaluation.
    :param weights_path: Path to the model weights. Optional since inside the training pipeline there are no weights to
    be loaded.
    :return: The classification model and the device, which are being used for training/evaluation.
    """
    # Sets the device for training
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice used for training/evaluation: {device}')
    print(f'\nLoading {model_name} model (and weights).')

    # Loads the selected classification model and sets the amount of output parameters to 43 (amount of GTSRB classes)
    model = models[model_name]  # Instantiates the model

    # Sets the amount of output parameters for the VGG models
    if 'VGG' in model_name:
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    # Sets the amount of output parameters for the InceptionV3 model
    elif model_name == 'InceptionV3':
        model.AuxLogits.fc = nn.Linear(768, config['num_classes'])
        model.fc = nn.Linear(2048, config['num_classes'])
    # Sets the amount of output parameters for the ResNet models (also includes ResNeXt) and adds a dropout layer
    elif 'Res' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_ftrs, config['num_classes'])
        )

    model = nn.DataParallel(model)  # Sets the model to work in parallel on all available GPUs

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=device))  # Loads the model weights
    model = model.to(device)  # Moves the model to the device
    print(model_name + ' has been loaded! \n')

    return model, device
def PCL_get_custom_label(y, dataset):

    y_new=np.zeros(y.size(0))
    for i in range(y.size(0)):
        if dataset=='gtsrb':
            y_new[i]=PCL_custom_labels_gtsrb[y[i].item()]
        elif dataset=='imagenet':
            y_new[i]=PCL_custom_labels_imagenet[y[i].item()]
        elif dataset=='cure_tsd':
            y_new[i] = PCL_custom_labels_cure_tsd[y[i].item()]
    return torch.from_numpy(y_new).cuda()

def PCL_gen_prox_opt(cfg, size):
    cluster={
            'criterion_prox_1':Proximity(num_classes=size, feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']], use_gpu=True),
            'criterion_prox_2':Proximity(num_classes=size, feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']], use_gpu=True),
            'criterion_conprox_1': Con_Proximity(num_classes=size, feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']],use_gpu=True),
            'criterion_conprox_2': Con_Proximity(num_classes=size, feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']],use_gpu=True)
        }
    cluster['optimizer_prox_1'] = torch.optim.SGD(cluster['criterion_prox_1'].parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    cluster['optimizer_prox_2'] = torch.optim.SGD(cluster['criterion_prox_2'].parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    cluster['optimizer_conprox_1'] = torch.optim.SGD(cluster['criterion_conprox_1'].parameters(), lr=0.0001, weight_decay=cfg['weight_decay'])
    cluster['optimizer_conprox_2'] = torch.optim.SGD(cluster['criterion_conprox_2'].parameters(), lr=0.0001, weight_decay=cfg['weight_decay'])

    return cluster

def PCL_custom_proxs(config):
    if config['dataset']=='gtsrb':
        cluster_prox={
            0:{
                'classes': [0, 1, 2, 3, 4, 5, 6, 7, 8],
                'prox':PCL_gen_prox_opt(config, 9)
            },
            1: {
                'classes': [9,10,15,16],
                'prox': PCL_gen_prox_opt(config, 4)
            },
            2: {
                'classes': [11,18,19,20,21,22,23,24,25,26,27,28,29,30,31],
                'prox': PCL_gen_prox_opt(config, 15)
            },
            3: {
                'classes': [32,41,42],
                'prox': PCL_gen_prox_opt(config, 3)
            },
            4: {
                'classes': [33,34,35,36,37,38,39,40],
                'prox': PCL_gen_prox_opt(config, 8)
            }
        }
    elif config['dataset']=='imagenet':
        cluster_prox = {
            0: {
                'classes': [0, 1, 2],
                'prox': PCL_gen_prox_opt(config, 3)
            },
            1: {
                'classes': [3,4,5,6],
                'prox': PCL_gen_prox_opt(config, 4)
            },
            2: {
                'classes': [7,8,9,10],
                'prox': PCL_gen_prox_opt(config, 4)
            },
            3: {
                'classes': [11,12,13,14],
                'prox': PCL_gen_prox_opt(config, 4)
            }
        }
    elif config['dataset']=='cure_tsd':
        cluster_prox = {
            0: {
                'classes': [1, 2],
                'prox': PCL_gen_prox_opt(config, 2)
            },
            1: {
                'classes': [3,4],
                'prox': PCL_gen_prox_opt(config, 2)
            },
            2: {
                'classes': [8,9,10],
                'prox': PCL_gen_prox_opt(config, 3)
            }
        }
    return cluster_prox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/train_config_custom_prox.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    args = parser.parse_args()

    # Reads the yaml file, which contains the train parameters
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    print(f"Initiating GTSRB Training for the {cfg['model_name']} model.")

    # Defines the training session name and creates a new sub-folder for storing the training results and log files
    session_name = f"{cfg['model_name']}_{cfg['training_method']}"
    session_path = f"{cfg['output_path']}/{session_name}"
    os.makedirs(session_path)

    # Displays all parameters for the current training session and creates a copy of the respective yaml file
    shutil.copyfile(args.config, f"{session_path}/train_config_{cfg['model_name']}.yaml")
    print('\nPARAMETERS')
    for param in cfg.keys():
        print('    ' + param + ' = ' + str(cfg[param]))

    torch.manual_seed(cfg['seed'])  # Sets the random seed for torch

    model, device = load_model(cfg['model_name'], cfg['device_ids'], cfg)  # Loads the classification model and the device

    # Loads the train and validation data (train data is being augmented)
    train_loader, val_loader = \
        get_train_loaders(cfg['data_path'], cfg['samples_per_class'], cfg['batch_size'], cfg['num_workers'],cfg['num_classes'], device)

    criterion = nn.CrossEntropyLoss()  # Sets the loss function for training the model to cross entropy loss


    # PCL Trainig MSPCL
    mspcl_criterion_prox_1 = Proximity(num_classes=9, feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']], use_gpu=True)
    mspcl_criterion_prox_2 = Proximity(num_classes=9, feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']], use_gpu=True)

    mspcl_criterion_conprox_1 = Con_Proximity(num_classes=9, feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']], use_gpu=True)
    mspcl_criterion_conprox_2 = Con_Proximity(num_classes=9, feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']], use_gpu=True)
    mspcl_custom_clusters=PCL_custom_proxs(cfg)

    # PCL Trainig WPCL
    wpcl_criterion_prox_1 = Proximity(num_classes=cfg['num_classes'], feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']],
                           use_gpu=True)
    wpcl_criterion_prox_2 = Proximity(num_classes=cfg['num_classes'], feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']],
                           use_gpu=True)

    wpcl_criterion_conprox_1024 = Con_Proximity(num_classes=cfg['num_classes'],
                                           feat_dim=cfg['cpl_crit_1_sizes'][cfg['model_name']], use_gpu=True,
                                           dataset=cfg['dataset'])
    wpcl_criterion_conprox_256 = Con_Proximity(num_classes=cfg['num_classes'],
                                          feat_dim=cfg['cpl_crit_2_sizes'][cfg['model_name']], use_gpu=True,
                                          dataset=cfg['dataset'])

    # Sets the optimizer as Stochastic Gradient Descent
    if(cfg['optimizer']) == 'SGD':
        optimizer = optim.SGD(model.parameters(), cfg['learning_rate'], momentum=cfg['momentum'],
                              weight_decay=cfg['weight_decay'])
    # Sets the optimizer as Adam
    elif(cfg['optimizer']) == 'Adam':
        optimizer = optim.Adam(model.parameters(), cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    # PCL Trainig MSPCL
    mspcl_optimizer_prox_1 = torch.optim.SGD(mspcl_criterion_prox_1.parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    mspcl_optimizer_prox_2 = torch.optim.SGD(mspcl_criterion_prox_2.parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    mspcl_optimizer_conprox_1 = torch.optim.SGD(mspcl_criterion_conprox_1.parameters(), lr=0.0001, weight_decay=cfg['weight_decay'])
    mspcl_optimizer_conprox_2 = torch.optim.SGD(mspcl_criterion_conprox_2.parameters(), lr=0.0001, weight_decay=cfg['weight_decay'])

    # PCL Trainig WPCL
    wpcl_optimizer_prox_1024 = torch.optim.SGD(wpcl_criterion_prox_1.parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    wpcl_optimizer_prox_256 = torch.optim.SGD(wpcl_criterion_prox_2.parameters(), lr=0.5, weight_decay=cfg['weight_decay'])
    wpcl_optimizer_conprox_1024 = torch.optim.SGD(wpcl_criterion_conprox_1024.parameters(), lr=0.0001,
                                             weight_decay=cfg['weight_decay'])
    wpcl_optimizer_conprox_256 = torch.optim.SGD(wpcl_criterion_conprox_256.parameters(), lr=0.0001,
                                            weight_decay=cfg['weight_decay'])

    # Instantiates TensorBoard SummaryWriters for logging training results to TensorBoard
    train_writer = SummaryWriter(session_path + '/TensorBoard_Logs/train')
    val_writer = SummaryWriter(session_path + '/TensorBoard_Logs/val')

    # Initiates the training process
    print(f"Initiating the training for {cfg['num_epochs']} epochs.")
    best_train_acc, best_train_loss, best_val_acc, best_val_loss = \
        train_model(
            cfg['num_epochs'],
            cfg['patience'],
            model,
            cfg['model_name'],
            criterion,
            optimizer,
            train_loader,
            val_loader,
            session_path,
            train_writer,
            val_writer,
            mspcl_optimizer_prox_2,
            mspcl_optimizer_prox_1,
            mspcl_optimizer_conprox_2,
            mspcl_optimizer_conprox_1,
            mspcl_criterion_prox_2,
            mspcl_criterion_prox_1,
            mspcl_criterion_conprox_2,
            mspcl_criterion_conprox_1,
            mspcl_custom_clusters,
            wpcl_optimizer_prox_256,
            wpcl_optimizer_prox_1024,
            wpcl_optimizer_conprox_256,
            wpcl_optimizer_conprox_1024,
            wpcl_criterion_prox_2,
            wpcl_criterion_prox_1,
            wpcl_criterion_conprox_256,
            wpcl_criterion_conprox_1024,
            cfg
        )
    print('Training finished.')

    # Logs the training hyperparameters and the final training results to TensorBoard
    #train_writer.add_hparams(cfg, {'Final_Train_Results/Train_Accuracy': best_train_acc,
    #                               'Final_Train_Results/Train_Loss': best_train_loss,
    #                               'Final_Train_Results/Val_Accuracy': best_val_acc,
    #                               'Final_Train_Results/Val_Loss': best_val_loss})
    #train_writer.close()


if __name__ == '__main__':
    main()
