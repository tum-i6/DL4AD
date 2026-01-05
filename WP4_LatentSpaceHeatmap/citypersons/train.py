import argparse
from datetime import datetime
import math
import os
import shutil
import sys
import torch
import yaml
from data import load_dataloaders
import utils
from evaluation import evaluate_model
from torch.utils.tensorboard import SummaryWriter
from models import load_model


def train_model(cfg, model, contra_weight, device, optimizer, lr_scheduler, train_loader, val_loader, session_path, starting_epoch, writer):
    """
    Optimizes the model weights for several epochs using the dataset from the train_loader. The model's prediction
    performance is evaluated after each epoch, using the validation dataset. Finally, a checkpoint file is stored after
    each epoch containing the model weights, optimizer state, learning rate scheduler state, the model's prediction
    performance on the validation set, the current epoch value and the average prediction losses during that epoch.

    :param cfg: A configuration dictionary containing the amount of epochs for training.
    :param model: The detection model, which should be optimized.
    :param contra_weight: the weighting factor before the contrastive loss term
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
        train_metrics = train_one_epoch(model, contra_weight, optimizer, train_loader, device, epoch, print_freq=10)
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
                # update learning rate
                writer.add_scalar("training/learning rate", train_metrics.meters[meter].global_avg, epoch)
                continue  # Skip the learning rate metric
            # update loss
            writer.add_scalar(f"training/{meter}", train_metrics.meters[meter].global_avg, epoch)
            checkpoint['losses'][meter] = train_metrics.meters[meter].global_avg

        # log the training info regarding the contrastive part
        writer.add_scalar(f'training/weighting factor of contrastive loss', contra_weight, epoch)

        # log the evaluation infos
        writer.add_scalars(
            f'Eval/metrics',
            {
                'AP coco': eval_results['coco_ap'],
                'AP pascal': eval_results['pascal_voc_ap'],
                'LAMR': eval_results['lamr']
            },
            epoch

        )
        writer.close()

        # save the current epoch
        torch.save(checkpoint, f'{session_path}/checkpoint_epoch_{epoch}.pth')


def train_one_epoch(model, contra_weight, optimizer, data_loader, device, epoch, print_freq):
    """
    Iterates over all sample images from the data_loader (one epoch) and uses the optimizer to optimize the model
    weights accordingly. Logs the model's train performance every print_freq amount of batches to the console.

    :param model: The detection model, which should be optimized.
    :param contra_weight: the weighting factor before the contrastive loss term
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
    with torch.autograd.set_detect_anomaly(True):
        for images, targets in metric_logger.log_every(data_loader, print_freq, header=f"Epoch: [{epoch}]"):
            # Moves the image samples and target tensors to the GPU device
            images = [image.to(device) for image in images]
            # prepare the targets
            if 'masks' in targets[0]:  # instance seg
                targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore', 'masks')} for t in targets]
            else:  # object det
                targets = [{key: t[key] for key in ('boxes', 'labels', 'area', 'ignore')} for t in targets]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # targets are a list of dicts

            # Infers the images into the model and computes the prediction losses
            loss_dict = model(images, targets)
            # initialize the losses
            losses = 0
            # sum up the individual losses
            for item in loss_dict:
                if item != 'contrastive_loss':
                    losses += loss_dict[item]
                else:
                    losses += contra_weight * loss_dict[item]  # weighting factor only for the new loss term

            # Checks whether the loss is infinite to stop the training
            if not math.isfinite(losses.item()):  # tensor.item()
                print(f"Loss is {losses.item()}, stopping training")
                print(loss_dict)
                sys.exit(1)

            # Backpropagates the prediction loss and optimizes the model weights. Mini-batch training
            optimizer.zero_grad()
            losses.backward()

            # check if the gradients are valid
            grad_is_exploding = torch.stack([torch.isnan(p.grad).any() for p in model.backbone.parameters()]).any()
            if grad_is_exploding.item():
                print(f'Nan values are found in the Gradients of backbone')

            # update the weights
            optimizer.step()

            # check if there is any nan value in the model
            is_nan = torch.stack([torch.isnan(p).any() for p in model.backbone.parameters()]).any()
            if is_nan.item():
                print(f'Nan values are found at the backbone')

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
    model weights and the test set prediction performance inside the two new checkpoints namely final_model_coco.pth (
    epoch with the highest coco ap value) and final_model_f1.pth (epoch with the highest f1 score).

    :param cfg: A configuration dictionary containing the amount of epochs for training.
    :param model: The detection model, which should be evaluated.
    :param device: The device used during training.
    :param test_loader: A dataloader, which holds the test dataset.
    :param session_path: The training session path, where each checkpoint file is stored.
    """
    print('\n\n--> Start Final Evaluation <--\n')

    # Determines the checkpoint file, which has achieved the best prediction performance on the validation dataset
    best_performance = 0
    best_performance_coco_ap = 0

    losses = {}  # Used for storing the loss values from all epochs
    val_performance = {}  # Used for storing the val_performance dictionaries from all epochs
    for epoch in range(cfg['num_epochs']):
        # ----------------find best f1----------------
        checkpoint = torch.load(f'{session_path}/checkpoint_epoch_{epoch}.pth')
        for conf_thr in checkpoint['val_performance']['eval_results']:
            if checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1'] is None:
                continue
            else:
                # Checks the F1-score for a specific confidence threshold and an IoU threshold of 0.25 (main metric)
                if checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1'] > best_performance:
                    best_performance = checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1']
                    best_conf_thr = conf_thr
                    best_checkpoint = f'checkpoint_epoch_{epoch}.pth'
                    best_epoch = epoch

        # ----------------find best ap----------------
        if checkpoint['val_performance']['coco_ap'] > best_performance_coco_ap:
            best_performance_coco_ap = checkpoint['val_performance']['coco_ap']
            best_checkpoint_coco_ap = f'checkpoint_epoch_{epoch}.pth'
            best_epoch_coco_ap = epoch

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

    # ----------------evaluate best f1 epoch----------------
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
    torch.save(checkpoint, f'{session_path}/final_model_f1.pth')

    # Prints the final evaluation results with respect to PDSM to the console
    utils.print_test_evaluation_report(f'{session_path}/final_model_f1.pth')

    # ----------------evaluate best coco epoch----------------
    # Loads the model weights from the best performing checkpoint file
    checkpoint = torch.load(f'{session_path}/{best_checkpoint_coco_ap}')
    model.load_state_dict(checkpoint['model'])

    # Evaluates the model weights from the checkpoint file on the test set
    eval_results = evaluate_model(model, test_loader, device)

    # Stores the final model weights and test set prediction performance inside a dictionary
    checkpoint = {
        'model': model.state_dict(),
        'val_performance': val_performance,
        'test_performance': eval_results,
        'train_losses': losses,
        'final_epoch': best_epoch_coco_ap,
        'conf_thr': '0.5'
    }
    torch.save(checkpoint, f'{session_path}/final_model_coco.pth')

    # Prints the final evaluation results with respect to PDSM to the console
    utils.print_test_evaluation_report(f'{session_path}/final_model_coco.pth')


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
    if (cfg['optimizer']) == 'SGD':
        optimizer = torch.optim.SGD(
            trainable_params,
            cfg['learning_rate'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay']
        )
    # Sets the optimizer as Adam
    elif (cfg['optimizer']) == 'Adam':
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

        # ------------contrastive loss weight------------
        contra_weight = torch.tensor(cfg['contrastive_loss_weighting_factor'])

        # Determines at which epoch the training should resume
        last_epoch = 0
        train_files = os.listdir(args.resume)
        for file in train_files:
            if file.startswith('checkpoint'):
                epoch_checkpoint = int(file.split('_')[-1][:-4])
                if epoch_checkpoint > last_epoch:
                    last_epoch = epoch_checkpoint
        if last_epoch + 1 == cfg['num_epochs']:
            print(f'Training cannot be resumed, since it already finished at epoch {last_epoch} (counting from 0).')
            sys.exit(1)
    else:
        # Reads the yaml file, which contains the train parameters
        with open(args.config) as yaml_file:
            cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

        # ------------contrastive loss weight------------
        contra_weight = torch.tensor(cfg['contrastive_loss_weighting_factor'])

        # Defines the training session name and creates a new sub-folder for storing the training results and logs
        session_name = \
            f"{str(datetime.now())[:-7].replace(' ', '_').replace(':', '-')}_{cfg['dataset']}_{cfg['model_name']}"
        session_path = f"{cfg['output_path']}/{session_name}_weight_{int(contra_weight.item()*100)}"
        os.makedirs(session_path)

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
        print(f'Model is resumed from the latest {last_epoch} epoch')

        starting_epoch = last_epoch + 1  # Sets the new starting epoch
        session_path = args.resume  # Sets the session path from which the training resumes
    else:
        starting_epoch = 0  # Sets the baseline starting epoch

    # Instantiates TensorBoard SummaryWriters for logging training results to TensorBoard
    writer = SummaryWriter(session_path + '/TensorBoard_Logs')

    # Initiates the training of the detection model
    train_model(cfg, model, contra_weight.item(), device, optimizer, lr_scheduler,
                train_loader, val_loader, session_path, starting_epoch, writer)

    # Initiates the evaluation of the detection model
    evaluate_train_session(cfg, model, device, test_loader, session_path)


if __name__ == '__main__':
    main()
