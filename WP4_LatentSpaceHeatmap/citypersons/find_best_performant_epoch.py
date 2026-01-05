import os
from data import load_dataloaders
import torch
import yaml
from models import load_model
from evaluation import evaluate_model
import utils
import tqdm
from lamr_eval import eval_demo


def evaluate_train_session(cfg, model, device, test_loader, session_path, n_epoch):  # cfg, model, device, test_loader,
    """
    Finds the checkpoint file (inside the session_path) with the best performing model weights on the validation
    dataset. Loads the checkpoint file and evaluates the model's prediction performance on the test dataset. Stores the
    model weights and the test set prediction performance inside a new checkpoint file called final_model.pth.

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
    for epoch in range(n_epoch):
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

    # # ----------------evaluate best f1 epoch----------------
    # # Loads the model weights from the best performing checkpoint file
    # checkpoint = torch.load(f'{session_path}/{best_checkpoint}')
    # model.load_state_dict(checkpoint['model'])
    #
    # # Evaluates the model weights from the checkpoint file on the test set
    # eval_results = evaluate_model(model, test_loader, device)
    #
    # # Stores the final model weights and test set prediction performance inside a dictionary
    # checkpoint = {
    #     'model': model.state_dict(),
    #     'val_performance': val_performance,
    #     'test_performance': eval_results,
    #     'train_losses': losses,
    #     'final_epoch': best_epoch,
    #     'conf_thr': best_conf_thr
    # }
    # print(f'best f1 {round(best_performance, 3)} found at {best_epoch} epoch')
    # torch.save(checkpoint, f'{session_path}/final_model_f1.pth')
    #
    # # Prints the final evaluation results with respect to PDSM to the console
    # utils.print_test_evaluation_report(f'{session_path}/final_model_f1.pth')

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
    print(f'best coco {round(best_performance_coco_ap, 3)} found at {best_epoch_coco_ap} epoch')
    torch.save(checkpoint, f'{session_path}/final_model_coco.pth')

    # Prints the final evaluation results with respect to PDSM to the console
    utils.print_test_evaluation_report(f'{session_path}/final_model_coco.pth')


def main():
    # get config file
    cfg_path = '.config/eval_config.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # session path
    session_path = '/'.join(cfg['pretrained_weights_path'].split('/')[:-1])

    # Loads the train, validation and test dataloader instances
    _, _, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # number epoch
    n_epoch = 0
    for epoch_path in os.listdir(session_path):
        if 'checkpoint_epoch' in epoch_path:
            n_epoch += 1
    print(f'Current session has {n_epoch} epochs')

    evaluate_train_session(cfg, model, device, test_loader, session_path, n_epoch)  # cfg, model, device, test_loader,


if __name__ == '__main__':
    main()
