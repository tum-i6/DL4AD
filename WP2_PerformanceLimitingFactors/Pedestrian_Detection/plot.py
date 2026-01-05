import math
import matplotlib.pyplot as plt
import torch


def plot_training_convergence(checkpoint_path, plot_title='Training Convergence', destination=None):
    """
    Loads a final_model.pth checkpoint file to extract the training performance values and plots them. The training loss
    is plotted against the best performing F1-score and the corresponding precision, recall and confidence thresholds on
    the validation dataset over all training epochs. Additionally, the final precision, recall and F1-score on the test
    dataset are marked within the plot.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    :param plot_title: A string that specifies the plot title.
    :param destination: A string path, that specifies the destination for storing the plot. Please also specify the
    file name and do not forget to add '.png' at the end.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    train_losses = checkpoint['train_losses']['loss']  # A list of training loss values over all epochs
    final_conf_thr = checkpoint['conf_thr']  # The final model specific confidence threshold
    # Stores the best performing F1-scores and the corresponding precision, recall and confidence threshold values
    # over all epochs
    precision_values = []
    recall_values = []
    f1_values = []
    best_conf_thrs = []
    for epoch in checkpoint['val_performance']:
        best_f1 = 0
        for conf_thr in checkpoint['val_performance'][epoch]['eval_results']:
            if checkpoint['val_performance'][epoch]['eval_results'][conf_thr]['0.25']['F1'] > best_f1:
                best_f1 = checkpoint['val_performance'][epoch]['eval_results'][conf_thr]['0.25']['F1']
                best_conf_thr = conf_thr
        precision_values.append(checkpoint['val_performance'][epoch]['eval_results'][best_conf_thr]['0.25']['precision'])
        recall_values.append(checkpoint['val_performance'][epoch]['eval_results'][best_conf_thr]['0.25']['recall'])
        f1_values.append(checkpoint['val_performance'][epoch]['eval_results'][best_conf_thr]['0.25']['F1'])
        best_conf_thrs.append(float(best_conf_thr))
    # Test precision, recall and F1-score for the best performing epoch
    test_precision = checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['precision']
    test_recall = checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['recall']
    test_f1 = checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['F1']

    # Generates the plot
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    val_precision_plt = ax1.plot(precision_values, color='#1f77b4', label='Validation Set Precision@0.25IoU')
    val_recall_plt = ax1.plot(recall_values, color='#ff7f04', label='Validation Set Recall@0.25IoU')
    val_f1_plt = ax1.plot(f1_values, color='#2ca02c', label='Validation Set F1-Score@0.25IoU')
    test_precision_plt = ax1.plot(checkpoint['final_epoch'], test_precision, marker='*', markersize=10,
                              color='#1f77b4', label='Test Set Precision@0.25IoU')
    test_recall_plt = ax1.plot(checkpoint['final_epoch'], test_recall, marker='*', markersize=10,
                           color='#ff7f04', label='Test Set Recall@0.25IoU')
    test_f1_plt = ax1.plot(checkpoint['final_epoch'], test_f1, marker='*', markersize=10,
                       color='#2ca02c', label='Test Set F1-Score@0.25IoU')
    conf_thr_plt = ax1.plot(best_conf_thrs, linestyle='dashed', color='black', label='Confidence Threshold')
    # Scale the left y-axis from 0 to 1
    plt.yticks([i/10 for i in range(11)])
    ax1.set_ylim([0, 1])
    ax1.set_ylabel('Precision / Recall / F1-Score / Confidence Threshold')

    # Plots the train loss over all epochs on the right y-axis
    ax2 = ax1.twinx()
    train_loss_plt = ax2.plot(train_losses, linestyle='dashed', color='r', label='Train Set Loss')
    # Scale the loss axis from 0 to rounded maximum loss value
    ax2.set_ylim([0, math.ceil(max(train_losses))])
    ax2.set_ylabel('Loss')

    # Sets up the x-axis
    ax1.set_xlim([0, len(train_losses)-1])
    ax1.set_xlabel('Epochs')

    # Add a legend under the plot
    lns = val_precision_plt + val_recall_plt + val_f1_plt + conf_thr_plt + \
          test_precision_plt + test_recall_plt + test_f1_plt + train_loss_plt
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(bottom=0.3)

    # Activates the plot grid and sets the title
    ax1.grid()
    plt.title(plot_title)

    # Either shows the plot directly or stores it at the specified destination path
    if destination:
        plt.savefig(destination)
    else:
        plt.show()


def plot_precision_recall_f1(checkpoint_path, iou_thr=0.5, plot_title='Precision-Recall-F1@0.5IoU', destination=None):
    """
    Loads a final_model.pth checkpoint file to extract the precision, recall & F1-score and plots them.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    :param iou_thr: A float number that specifies the IoU threshold for the plot.
    :param plot_title: A string that specifies the plot title.
    :param destination: A string path, that specifies the destination for storing the plot. Please also specify the
    file name and do not forget to add '.png' at the end.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    precisions = []  # A list of test set precision values over all confidence thresholds
    recalls = []  # A list of test set recall values over all confidence thresholds
    f1s = []  # A list of test set F1-scores over all confidence thresholds
    conf_thresholds = []  # A list of confidence thresholds
    final_conf_thr = checkpoint['conf_thr']  # The final model specific confidence threshold
    for conf_threshold in checkpoint['test_performance']['eval_results']:
        precisions.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['precision'])
        recalls.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['recall'])
        f1s.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['F1'])
        conf_thresholds.append(float(conf_threshold))

    # Generates the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(conf_thresholds, precisions, label='Precision')
    ax.plot(conf_thresholds, recalls, label='Recall')
    ax.plot(conf_thresholds, f1s, label='F1-Score')
    ax.axvline(x=float(final_conf_thr), linestyle='dashed', color='black', label='Final Confidence Threshold')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_ylabel('Precision / Recall / F1-Score')
    ax.set_xlabel('Confidence Threshold')
    ax.set_title(plot_title)
    # Add a legend under the plot
    plt.legend(['Precision', 'Recall', 'F1-Score', 'Final Confidence Threshold'],
               loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    fig.subplots_adjust(bottom=0.2)
    ax.grid()

    # Either shows the plot directly or stores it at the specified destination path
    if destination:
        plt.savefig(destination)
    else:
        plt.show()


def plot_precision_recall_curve(checkpoint_path, iou_thr=0.5, plot_title='Precision-Recall@0.5IoU', destination=None):
    """
    Loads a final_model.pth checkpoint file to extract the precision & recall values and plots them.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    :param iou_thr: A float number that specifies the IoU threshold for the precision-recall curve.
    :param plot_title: A string that specifies the plot title.
    :param destination: A string path, that specifies the destination for storing the plot. Please also specify the
    file name and do not forget to add '.png' at the end.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    precisions = []  # A list of test set precision values over all confidence thresholds
    recalls = []  # A list of test set recall values over all confidence thresholds
    for conf_threshold in checkpoint['test_performance']['eval_results']:
        precisions.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['precision'])
        recalls.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['recall'])

    # Generates the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recalls, precisions)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.set_title(plot_title)
    ax.grid()

    # Either shows the plot directly or stores it at the specified destination path
    if destination:
        plt.savefig(destination)
    else:
        plt.show()


def plot_recall_iou_curve(checkpoint_path, plot_title='Recall-IoU', destination=None):
    """
    Loads a final_model.pth checkpoint file to extract the recall & IoU values and plots them.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    :param plot_title: A string that specifies the plot title.
    :param destination: A string path, that specifies the destination for storing the plot. Please also specify the
    file name and do not forget to add '.png' at the end.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    recalls = []  # A list of test set recall values over all IoU thresholds
    ious = []  # A list of IoU thresholds
    conf_thr = checkpoint['conf_thr']
    for iou_threshold in checkpoint['test_performance']['eval_results'][conf_thr]:
        recalls.append(checkpoint['test_performance']['eval_results'][conf_thr][iou_threshold]['recall'])
        ious.append(float(iou_threshold))

    # Generates the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ious, recalls)
    ax.set_ylim([0, 1])
    ax.set_xlim([0.25, 1])  # IoU thresholds start from 0.25
    ax.set_ylabel('Recall')
    ax.set_xlabel('IoU')
    ax.set_title(plot_title)
    ax.grid()

    # Either shows the plot directly or stores it at the specified destination path
    if destination:
        plt.savefig(destination)
    else:
        plt.show()


def plot_mr_fppi_curve(checkpoint_path, iou_thr=0.5, plot_title='MR-FPPI@0.5IoU', destination=None):
    """
    Loads a final_model.pth checkpoint file to extract the MR & FPPI values and plots them.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    :param iou_thr: A float number that specifies the IoU threshold for the MR-FPPI curve.
    :param plot_title: A string that specifies the plot title.
    :param destination: A string path, that specifies the destination for storing the plot. Please also specify the
    file name and do not forget to add '.png' at the end.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    mr = []  # A list of test set MR values over all confidence thresholds
    fppi = []  # A list of test set FPPI values over all confidence thresholds
    for conf_threshold in checkpoint['test_performance']['eval_results']:
        mr.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['MR'])
        fppi.append(checkpoint['test_performance']['eval_results'][conf_threshold][str(iou_thr)]['FPPI'])

    # Generates the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fppi, mr)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim([10**-2, 10**0])
    ax.set_xlim([10**-4, 10**1])
    ax.set_yticks(
        [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    ax.set_ylabel('Miss Rate')
    ax.set_xlabel('False Positives per Image')
    ax.set_title(plot_title)
    ax.grid()

    # Either shows the plot directly or stores it at the specified destination path
    if destination:
        plt.savefig(destination)
    else:
        plt.show()
