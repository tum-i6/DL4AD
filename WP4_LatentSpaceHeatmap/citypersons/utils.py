from collections import defaultdict, deque
import datetime
import time
from sklearn.metrics import auc
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def get_edge_magnitude(img):
    """
    Uses the sobel filter kernel to detect horizontal and vertical edges in a grayscale image. The horizontal and
    vertical derivatives are then combined yielding the gradient magnitude, which contains the final edge detections.

    :param img: A grayscale numpy image.
    :return: A single-channel numpy image holding the edge information.
    """
    # Calculates the derivatives dx and dy using the sobel filtering
    dx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # Calculates the magnitude for the derivatives dx and dy
    magnitude = cv2.magnitude(dx, dy)
    return magnitude


def get_log_avg_miss_rate(eval_results, iou_threshold='0.5'):
    """
    Computes the log average miss rate for a specific IoU threshold.

    :param eval_results: A dictionary that contains various confidence thresholds as keys. The value of each confidence
    threshold key is a dictionary that contains various IoU thresholds as keys. The value of each Iou threshold key is
    another dictionary, which contains the respective MR (miss rate) and FPPI (false positives per image) values.
    :param iou_threshold: A string of a decimal number that specifies the IoU threshold.
    :return: The log average miss rate value as a float number.
    """
    miss_rates = []  # Stores the MR values for the MR-FPPI curve
    fppi_values = []  # Stores the FPPI values for the MR-FPPI curve
    # Iterates over multiple confidence thresholds and extracts the respective MR and FPPI values
    for conf_threshold in eval_results:
        miss_rates.append(eval_results[conf_threshold][iou_threshold]['MR'])
        fppi_values.append(eval_results[conf_threshold][iou_threshold]['FPPI'])
    miss_rates = np.array(miss_rates)
    fppi_values = np.array(fppi_values)

    refs = np.logspace(-2, 0, num=9)  # Defines the FPPI thresholds in log-space
    for i, fppi_thr in enumerate(refs):
        idx = np.argmax(fppi_values <= fppi_thr)
        refs[i] = miss_rates[idx]

    # # plot the curve
    # if iou_threshold == '0.5':
    #     plt.plot(np.logspace(-2, 0, num=9), refs, "--r*")
    #     plt.xscale('log', base=10)
    #     # plt.yscale('log', base=2)
    #     plt.xlabel("FPPI")
    #     plt.ylabel("Miss-rate")
    #     plt.ylim([0, 1.05])
    #     plt.title(f'MR vs. FPPI-Curve at {iou_threshold} IoU\nLAMR {np.exp(np.mean(np.log(refs))):.2f}')
    #     plt.show()
    try:
        LAMR=np.exp(np.mean(np.log(refs)))
    except ZeroDivisionError:
        LAMR=1.0
    return LAMR


def get_average_precisions(eval_results):
    """
    Computes the average precision values for multiple IoU thresholds (25 to 95 with a step-size of 5).

    :param eval_results: A dictionary that contains various confidence thresholds as keys. The value of each confidence
    threshold key is a dictionary that contains various IoU thresholds as keys. The value of each IoU threshold key is
    another dictionary, which contains the respective precision and recall values.
    :return: A dictionary that contains various IoU threshold as keys. The value of each IoU threshold key is the
    respective average precision value.
    """
    aps = {}  # Stores the average precision values over multiple IoU thresholds
    for iou_threshold in range(25, 96, 5):  # Iterates over multiple IoU thresholds
        iou_threshold = str(iou_threshold / 100)
        precisions = []  # Stores the precision values for the precision-recall curve
        recalls = []  # Stores the recall values for the precision-recall curve
        # Iterates over multiple confidence thresholds and extracts the respective precision and recall values
        for conf_threshold in eval_results:
            precisions.append(eval_results[conf_threshold][iou_threshold]['precision'])
            recalls.append(eval_results[conf_threshold][iou_threshold]['recall'])
        # Adds the final precision and recall values to ease the computation of the average precision
        precisions.append(1)
        recalls.append(0)
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        aps[iou_threshold] = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])  # Computes the average precision

    return aps

def get_ROC(eval_results,iou_threshold ="0.5" ):
    """
    Computes the average precision values for multiple IoU thresholds (25 to 95 with a step-size of 5).

    :param eval_results: A dictionary that contains various confidence thresholds as keys. The value of each confidence
    threshold key is a dictionary that contains various IoU thresholds as keys. The value of each IoU threshold key is
    another dictionary, which contains the respective precision and recall values.
    :return: A dictionary that contains various IoU threshold as keys. The value of each IoU threshold key is the
    respective average precision value.
    """
    aps = {}  # Stores the average precision values over multiple IoU thresholds
    conf_thresholds =[]
    precisions = []  # Stores the precision values for the precision-recall curve
    recalls = []  # Stores the recall values for the precision-recall curve
    # Iterates over multiple confidence thresholds and extracts the respective precision and recall values
    for conf_threshold in eval_results:
        precisions.append(eval_results[conf_threshold][iou_threshold]['precision'])
        recalls.append(eval_results[conf_threshold][iou_threshold]['recall'])
        conf_thresholds.append(conf_threshold)
    # Adds the final precision and recall values to ease the computation of the average precision
    precisions.append(1)
    recalls.append(0)
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    aps['conf_threshods'] = conf_thresholds
    aps['precisions'] =precisions.tolist()
    aps['recalls'] =recalls.tolist()
    auc_score = auc(recalls, precisions)
    aps['auc_score'] = round(auc_score, 3)
    return aps

def print_test_evaluation_report(checkpoint_path):
    """
    Loads a final_model.pth checkpoint file to extract the prediction performance values for the test & validation set
    and logs them to the console.

    :param checkpoint_path: String path to a final_model.pth checkpoint file.
    """
    # Loads the checkpoint file to extract the performance values
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    final_epoch = checkpoint['final_epoch']  # Epoch at which the highest F1-score on the validation set was achieved
    # Corresponding confidence threshold with whom the highest F1-score on the validation set was achieved
    final_conf_thr = checkpoint['conf_thr']

    # Uses the final_epoch and final_conf_thr values to extract the final precision, recall and F1-score for the val set
    val_prec = round(checkpoint['val_performance'][final_epoch]['eval_results'][final_conf_thr]['0.25']['precision'], 3)
    val_rec = round(checkpoint['val_performance'][final_epoch]['eval_results'][final_conf_thr]['0.25']['recall'], 3)
    val_f1 = round(checkpoint['val_performance'][final_epoch]['eval_results'][final_conf_thr]['0.25']['F1'], 3)
    # Extracts the final detection performance values for the test set
    coco_ap = checkpoint['test_performance']['coco_ap']
    pascal_voc_ap = checkpoint['test_performance']['pascal_voc_ap']
    lamr = checkpoint['test_performance']['lamr']
    custom_ap = checkpoint['test_performance']['custom_ap']
    custom_lamr = checkpoint['test_performance']['custom_lamr']
    precision = round(checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['precision'], 3)
    recall = round(checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['recall'], 3)
    f1 = round(checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['F1'], 3)
    fppi = round(checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['FPPI'], 3)
    mr = round(checkpoint['test_performance']['eval_results'][final_conf_thr]['0.25']['MR'], 3)
    inference_time = round(1 / checkpoint['test_performance']['model_time'], 3)

    # Logs the performance values to the console
    print('\nTest Set Prediction Performance Summary')
    print('Standard Metrics:')
    print(f'Average Precision           (AP)    @[ IoU=0.5:0.05:0.95  &  C=0.0:0.05:1.0 ] = {coco_ap}')
    print(f'Average Precision           (AP)    @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {pascal_voc_ap}')
    print(f'Log Average Miss Rate       (LAMR)  @[ IoU=0.5            &  C=0.0:0.05:1.0 ] = {lamr}')
    print('\nPedestrian Detection Safety Metric (PDSM):')
    print(f'Average Precision           (AP)    @[ IoU=0.25           &  C=0.0:0.05:1.0 ] = {custom_ap}')
    print(f'Log Average Miss Rate       (LAMR)  @[ IoU=0.25           &  C=0.0:0.05:1.0 ] = {custom_lamr}')
    print(f'Miss Rate                   (MR)    @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {mr}')
    print(f'False Positives per Image   (FPPI)  @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {fppi}')
    print(f'F1-Score                    (F1)    @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {f1}')
    print(f'Precision                   (P)     @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {precision}')
    print(f'Recall                      (R)     @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {recall}')

    print('\n\nValidation Set Prediction Performance Summary')
    print('Pedestrian Detection Safety Metric (PDSM):')
    print(f'F1-Score                    (F1)    @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {val_f1}')
    print(f'Precision                   (P)     @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {val_prec}')
    print(f'Recall                      (R)     @[ IoU=0.25           &  C={float(final_conf_thr):.2f}         ] = {val_rec}')

    print('\n\nInference Time:')
    print(f'{inference_time} FPS')
    print('\nTraining Duration:')
    print(f'{checkpoint["final_epoch"]} Epochs')


def collate_fn(batch):
    """
    Creates the batch of data samples and targets for the dataloader.

    :param batch: A list of tuples, where each tuple contains the image sample and the corresponding targets.
    :return: A tuple of two elements, where the first element contains a tuple of all image samples and the second
    element contains a tuple of the corresponding target labels.
    """
    return tuple(zip(*batch))


class SmoothedValue:
    """
    Class used for tracking a series of values for a single metric. Provides access to smoothed values over a window or
    the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """
        Instantiates the class for tracking a series of values for a single metric.

        :param window_size: The amount of previous values that should be cached for tracking.
        :param fmt: A string that defines, which value properties should be returned and how many decimal points they
        should contain.
        """
        if fmt is None:
            # Standard properties that are being tracked and returned (the median value of all cached values and the
            # global series average
            fmt = "{median:.4f} ({global_avg:.4f})"

        # Used for caching all the previous values up to the amount specified by the window_size
        self.deque = deque(maxlen=window_size)
        self.total = 0.0  # Used for summing up all values (used for the computing the global series average)
        self.count = 0  # Used for counting the amount of value updates (for computing the global series average)
        self.fmt = fmt

    def __str__(self):
        """
        :return: A string that contains specific value properties that have been specified by the fmt attribute.
        """
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )

    def update(self, value, n=1):
        """
        Adds the new value to the cache and updates the other value properties accordingly.

        :param value: A new integer or float value that is being assigned.
        :param n: A integer that defines how often the value should be added at once.
        """
        self.deque.append(value)  # Caches the new value
        self.count += n  # Updates the update counter
        self.total += value * n  # Updates the total sum of all values

    @property
    def median(self):
        """
        :return: The median value of all cached values, which have been tracked with respect to the window_size.
        """
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """
        :return: The average value of all cached values, which have been tracked with respect to the window_size.
        """
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """
        :return: The average value, with respect to all values that have been seen so far (the global series average).
        """
        return self.total / self.count

    @property
    def max(self):
        """
        :return: The maximum value of all cached values, which have been tracked with respect to the window_size.
        """
        return max(self.deque)

    @property
    def value(self):
        """
        :return: The last added value.
        """
        return self.deque[-1]


class MetricLogger:
    """
    Class used for tracking different sorts of metrics corresponding to a dataloader instance.
    """

    def __init__(self, delimiter="\t"):
        """
        Instantiates the class for tracking different sorts of metrics.

        :param delimiter: A string that defines the delimiter space between the various metric logs that are being
        printed to the console.
        """
        self.meters = defaultdict(SmoothedValue)  # Used as a placeholder for tracking different metrics
        self.delimiter = delimiter

    def __getattr__(self, attr):
        """
        Returns an instance of the class SmoothedValue, that is used for storing and managing a specific metric.

        :param attr: The string name of the metric.
        :return: The instance of the class SmoothedValue, that is used for storing and managing a specific metric.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """
        :return: A string that contains the name of each metric that is being tracked and its corresponding value. The
        metrics are separated within the string by the delimiter string.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def update(self, **kwargs):
        """
        Updates the metric values, that are being tracked.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()  # Extracts the value from the torch tensor
            assert isinstance(v, (float, int))
            self.meters[k].update(v)  # Updates the metric value

    def add_meter(self, name, meter):
        """
        Adds a new metric that should be tracked.

        :param name: String name of the metric.
        :param meter: An instance of the class SmoothedValue, used for storing and managing the metric value.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        Iterates over all batches from the iterable and yields them to the process that calls this function. Logs the
        specific metrics and batch processing information to the console every print_freq amount of batches.

        :param iterable: A dataloader instance which is being iterated over.
        :param print_freq: An integer that specifies the amount of processed batches after which the logs should be
        printed to the console every time.
        :param header: The leading header for the logs that are being printed to the console. Must be of type string.
        """
        i = 0  # Used for tracking the amount of processed batches
        start_time = time.time()  # Used for tracking how long it took to iterate over all batches
        end = time.time()  # Tracks how long it took to load and process one batch

        # Instantiates the tracking metrics
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        # Loads the logging text as a string value
        if not header:
            header = ""
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )

        MB = 1024.0 * 1024.0  # Amount of bytes in 1MB

        # Iterates over all batches
        for obj in iterable:
            data_time.update(time.time() - end)  # Tracks how long it took to load the batch
            yield obj  # Processes the batch
            iter_time.update(time.time() - end)  # Tracks how long it took to process the batch

            # Prints the logs to the console
            if i % print_freq == 0 or i == len(iterable) - 1:
                # Estimates the time it will take to iterate over the remaining batches
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                # Prints the logs to the console
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,  # Amount of processed batches
                            len(iterable),  # Total amount of batches
                            eta=eta_string,  # Estimated time to iterate over the remaining batches
                            meters=str(self),  # All model specific metrics that are being tracked
                            time=str(iter_time),  # The average time for processing a single batch
                            data=str(data_time),  # How long it took to load the batch
                            # The maximum amount of memory that was being used (in MB)
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(  # Same as above, just without the GPU memory allocation information
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()  # Tracks how long it took to load and process the batch

        # Computes the time it took to iterate over all batches and logs it to the console
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")
