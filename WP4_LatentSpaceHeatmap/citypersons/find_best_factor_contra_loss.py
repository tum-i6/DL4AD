import os
from matplotlib import pyplot as plt

from data import load_dataloaders
import torch
import yaml
from models import load_model
from evaluation import evaluate_model
import utils
import tqdm
import statsmodels.api as sm

colors_custom = ['#88CCEE', '#DDCC77', '#6e13a9', '#d0af0a', '#6699CC', '#44AA99', '#b07e13', '#88f5c5', '#b8c08c',
                 '#239e36', '#75092c', '#67117a']
markers = ['o', '+', '>', '*', 'X', 'x', 'H', 'h', '1', '2', '3', '4']


def pick_up_best_epoch(session_path, n_epoch):
    # Determines the checkpoint file, which has achieved the best prediction performance on the validation dataset
    best_performance_f1 = 0
    best_performance_ap = 0
    best_performance_lamr = 1

    losses = {}  # Used for storing the loss values from all epochs
    val_performance = {}  # Used for storing the val_performance dictionaries from all epochs

    for epoch in range(n_epoch):
        checkpoint = torch.load(f'{session_path}/checkpoint_epoch_{epoch}.pth')

        # ----------------find best f1----------------
        for conf_thr in checkpoint['val_performance']['eval_results']:
            # Checks the F1-score for a specific confidence threshold and an IoU threshold of 0.25 (main metric)
            if checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1'] > best_performance_f1:
                best_performance_f1 = checkpoint['val_performance']['eval_results'][conf_thr]['0.25']['F1']
                best_conf_thr_f1 = conf_thr
                best_checkpoint_f1 = f'checkpoint_epoch_{epoch}.pth'
                best_epoch_f1 = epoch
        # ----------------find best ap----------------
        if checkpoint['val_performance']['pascal_voc_ap'] > best_performance_ap:  # coco_ap
            best_performance_ap = checkpoint['val_performance']['pascal_voc_ap']
            best_checkpoint_ap = f'checkpoint_epoch_{epoch}.pth'
            best_epoch_ap = epoch
        # ----------------find best lamr----------------
        if checkpoint['val_performance']['lamr'] < best_performance_lamr:
            best_performance_lamr = checkpoint['val_performance']['lamr']
            best_checkpoint_lamr = f'checkpoint_epoch_{epoch}.pth'
            best_epoch_lamr = epoch

        del checkpoint

    return best_performance_f1, best_performance_ap, best_performance_lamr


def get_n_epoch(path):
    # number epoch
    n_epoch = 0
    for epoch_path in os.listdir(path):
        if 'checkpoint_epoch' in epoch_path:
            n_epoch += 1

    return n_epoch


def turning_plot(facor_list, metric_dict):
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)

    lns = []
    for idx, metric in enumerate(metric_dict):
        # plot the both F1 and AP lines
        plot = ax1.plot(facor_list, metric_dict[metric], f'{markers[idx]}--', markersize=10,
                        linewidth=1.5, color=colors_custom[idx], label=f'{metric}')

        # add values on
        for index in range(len(metric_dict[metric])):
            ax1.text(facor_list[index], metric_dict[metric][index], round(metric_dict[metric][index], 2), size=10)

        # baseline model
        ref = ax1.axhline(metric_dict[metric][0], lw=.5, color=colors_custom[idx], label=f'baseline {metric}')

        # scatter_plot = ax1.scatter(facor_list, metric_dict[metric], s=10, linewidth=1.5, color=colors_custom[idx],
        #                            marker=markers[idx], label=f'{metric}')
        # estimate the trend line
        # trend_line = sm.nonparametric.lowess(facor_list, metric_dict[metric], frac=1 / 5)
        # trend_line = ax1.plot(trend_line[:, 0], trend_line[:, 1], f'{markers[idx]}-',
        #                       color=colors_custom[idx],  label=f'{metric} trend line')
        # add legend
        lns.append(plot[0])
        lns.append(ref)
        # lns.append(trend_line[0])

    ax1.set_xlabel(f"weighting factor of contrastive loss")
    # ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Val-set evaluation')

    # # Plots the histogram values on the second y-axis
    # ax2 = ax1.twinx()
    # hist_bars = ax2.bar(x_axis_values, n_gts, width=0.007, color='#332288', alpha=0.3, label='PLF Density Histogram')
    # lns.append(hist_bars)
    # ax2.set_ylabel('Density Histogram')

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.subplots_adjust(bottom=0.1)
    # plt.grid(color='lightgray', axis='y', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def test_set_best_factors(path, test_loader, model, device):
    # placeholder for factor
    factor_list = []
    # metrics
    ap_coco_list, ap_pascal_list, lamr_list = [], [], []
    # custom metrics at 0.25 IoU
    custom_ap_list, custom_lamr_list = [], []

    pbar = tqdm.tqdm(sorted(os.listdir(path)))
    for weight in pbar:
        # full path including the current session
        full_folder_path = f'{path}/{weight}'

        if 'final_model.pth' not in os.listdir(full_folder_path):
            continue

        # weighting factor used in the current session
        factor = int(weight.split('_')[-1]) / 100
        factor_list.append(factor)
        pbar.set_description(f"Processing weighting factor for the test-set {factor}")

        # load the model achieved highest F1 score on val-set
        checkpoint = torch.load(f'{full_folder_path}/final_model.pth')

        # load the best model
        model.load_state_dict(checkpoint['model'])

        # start evaluation
        # eval_results = evaluate_model(model, test_loader, device)

        # add evaluation
        test_eval = checkpoint['test_performance']

        # update the test-set evaluation
        # checkpoint['test_performance'] = test_eval
        # torch.save(checkpoint, f'{full_folder_path}/final_model_new.pth')
        # get metric values
        ap_coco, ap_pascal, lamr = test_eval['coco_ap'], test_eval['pascal_voc_ap'], test_eval['lamr']
        custom_ap, custom_lamr = test_eval['custom_ap'], test_eval['custom_lamr']

        # append
        ap_coco_list.append(ap_coco)
        ap_pascal_list.append(ap_pascal)
        lamr_list.append(lamr)
        custom_ap_list.append(custom_ap)
        custom_lamr_list.append(custom_lamr)

    # dict to store results
    metric_dict = {
        'AP coco': ap_coco_list,
        'AP pascal': ap_pascal_list,
        'LAMR': lamr_list,
        # 'AP custom': custom_ap_list,
        # 'LAMR custom': custom_lamr_list,
    }

    # plot
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    lns = []

    for idx, metric in enumerate(metric_dict):
        # plot the both F1 and AP lines
        plot = ax1.plot(factor_list, metric_dict[metric], f'{markers[idx]}--', markersize=10,
                        linewidth=1.5, color=colors_custom[idx], label=f'{metric}')

        # add values on
        for index in range(len(metric_dict[metric])):
            ax1.text(factor_list[index], metric_dict[metric][index], round(metric_dict[metric][index], 3), size=10)

        # baseline model
        ref = ax1.axhline(metric_dict[metric][0], lw=.5, color=colors_custom[idx], label=f'baseline {metric}')

        # add legend
        lns.append(plot[0])
        lns.append(ref)

    ax1.set_xlabel(f"weighting factor of contrastive loss")
    # ax1.set_ylim([0, 1.03])
    ax1.set_ylabel('Val-set evaluation')

    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(metric_dict))
    fig.subplots_adjust(bottom=0.1)
    fig.suptitle(f'val-set weighting factor tuning plots')
    # plt.grid(color='lightgray', axis='y', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def val_set_best_factors(path, n_epoch):
    # placeholders
    factor_list = []  # for the used factors
    best_f1_list = []  # for the best f1
    best_ap_list = []  # for the best ap
    best_lamr_list = []  # for the best lmar

    pbar = tqdm.tqdm(sorted(os.listdir(path)))
    for weight in pbar:
        # full path including the current session
        full_folder_path = f'{path}/{weight}'

        # pass the incomplete training session
        if n_epoch != get_n_epoch(full_folder_path) and get_n_epoch(full_folder_path) != 50:
            continue

        # weighting factor used in the current session
        factor = int(weight.split('_')[-1]) / 100
        factor_list.append(factor)
        pbar.set_description(f"Processing weighting factor for the val-set {factor}")

        # eval the current sessions
        best_f1, best_ap, best_lamr = pick_up_best_epoch(full_folder_path, n_epoch)

        # append
        best_f1_list.append(best_f1)
        best_ap_list.append(best_ap)
        best_lamr_list.append(best_lamr)

    assert len(factor_list) == len(best_f1_list)

    metric_dict = {
        'F1': best_f1_list,
        'AP Pascal': best_ap_list,
        'LAMR': best_lamr_list
    }

    turning_plot(factor_list, metric_dict)


def main():
    # get config file
    cfg_path = '.config/eval_config.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the train, validation and test dataloader instances
    _, _, test_loader = load_dataloaders(cfg)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # path including the tuning training models
    path = '/path/to/output_dir/tuning_loss_weight/'

    # how many epochs are trained for each training session
    n_epoch = 30

    # best factors on the val-set
    # val_set_best_factors(path, n_epoch)

    # best factor on the test-set
    test_set_best_factors(path, test_loader, model, device)


if __name__ == '__main__':
    main()
