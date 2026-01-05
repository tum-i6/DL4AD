import pickle
import numpy as np
import torchvision.transforms
import torch
import tqdm
import yaml
from extract_activation_patches import load_trained_faster_rcnn
from Average_Precision import testset_dataloader_kia
from matplotlib import pyplot as plt
import matplotlib.patches as patches


def occluded_plot(image, target, feature_map, refactored_bboxes, outlier_model, outlier_results):
    """
    Plot a 2*2 figure: the top-left figure shows the input image along with the GT bboxes of the occluded pedestrian instance(s).
    The OR represnets Occlusion Ratio. The top-right figure shows the input image's feature maps. The down-left subfigure shows
    the outlier detections map by running the outlier detector on the whole feature maps. The down-right subfigure shows overlaying
    the binary outlier detections map on the feature maps.
    :param image: input image as tensor on CPU
    :param target: corresponding ground truth
    :param feature_map: feature maps obtained from the last conv layer of the encoder part
    :param refactored_bboxes: transformed coordinates of the bboxes at the space of feature maps
    :param outlier_model: pre-trained outlier detector
    :param outlier_results: predictions on the extracted activation patches
    """
    # for visualization
    names = {1: 'inlier',
             -1: 'outlier'}
    colors = {1: 'g',
              -1: 'r'}
    # turn the feature map into a numpy array
    feature_map_array = feature_map.cpu().numpy()
    c_f, h_f, w_f = feature_map.shape
    # creating a list to store all the activation splits
    extraction_list = []
    # splitting the whole feature maps
    for j in range(h_f):
        for k in range(0, w_f - 1, 2):
            extraction_list.append(feature_map_array[:, j:(j + 1), k:(k + 2)])
    # flattening the splits for outlier detection
    extraction_array = np.array(extraction_list).reshape(len(extraction_list), -1)
    # pass the feature matrix to the outlier detector
    y_pred = outlier_model.predict(extraction_array)
    # resize the outlier predictions to the same as activation
    resized_pred = np.array([])
    for y in y_pred:
        resized_pred = np.append(resized_pred, [y, y])
    resized_pred = resized_pred.reshape(h_f, w_f)

    # get the GT bboxes coordinates from the target
    tar_bboxes = target[0]['boxes'].clone()
    tar_bboxes[:, [2]] = tar_bboxes[:, [2]] - tar_bboxes[:, [0]]
    tar_bboxes[:, [3]] = tar_bboxes[:, [3]] - tar_bboxes[:, [1]]
    # get the transformed bboxes coordinates from the detections
    bboxes_vis = refactored_bboxes.clone()
    bboxes_vis[:, [0, 1]] = torch.floor(bboxes_vis[:, [0, 1]])
    bboxes_vis[:, [2, 3]] = torch.ceil(bboxes_vis[:, [2, 3]])
    bboxes_vis[:, [2]] = bboxes_vis[:, [2]] - bboxes_vis[:, [0]]
    bboxes_vis[:, [3]] = bboxes_vis[:, [3]] - bboxes_vis[:, [1]]
    # take the depth-averaged feature map
    visul_activation = feature_map
    max_act = torch.mean(visul_activation, dim=0)
    # plot
    plt.figure()
    f, ax = plt.subplots(2, 2)
    # the input image along with the GT bboxes of the occluded pedestrian instance(s)
    ax[0, 0].imshow(image[0].permute(1, 2, 0).cpu().numpy())
    ax[0, 0].set_title('Input image with GT(g) bboxes')
    ax[0, 0].axis('off')
    # add the GT bboxes
    for idx, box in enumerate(tar_bboxes):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[0, 0].add_patch(rect)
        # add the occlusion ratio info
        ratio = round(target[0]['occlusion_ratio_est'][idx].item(), 2)
        ax[0, 0].text(box[0] + 1, box[1], f'OR:{ratio}', fontsize=10, color='w')
    # input image's feature maps with the transformed GT bboxes
    ax[0, 1].imshow(max_act.cpu().numpy(), cmap='gray')
    ax[0, 1].set_title('feature maps (depth-wise averaged)')
    # add the bboxes
    for idx, box in enumerate(bboxes_vis):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[0, 1].add_patch(rect)
    ax[0, 1].axis('off')
    # the outlier detections map by running the outlier detector on the whole feature maps
    ax[1, 0].imshow(resized_pred, cmap='gray')
    ax[1, 0].set_title('Outlier detections map')
    ax[1, 0].axis('off')
    # overlaying the binary outlier detections map on the feature maps
    overlaying = (resized_pred + 1) / 2 * max_act.cpu().numpy()
    ax[1, 1].imshow(overlaying, cmap='gray')
    ax[1, 1].set_title('outlier detections of GT(g) bboxes')
    # add the bboxes
    for idx, box in enumerate(bboxes_vis):
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[1, 1].add_patch(rect)
        # plot the outlier detection results on the subfigure
        ax[1, 1].text(box[0], box[1], str(names[outlier_results[idx]]), fontsize=10,
                      color=colors[outlier_results[idx]])
    plt.axis('off')
    plt.show()


def main():
    # Reads the yaml file, which contains the parameters for loading the training dataset
    cfg_path = '/path/to/testset_outlier_detector.yaml'
    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)
    # load the dataloader for KIA test set
    test_loader = testset_dataloader_kia(cfg)
    # load the pre-trained model
    weight_path = '/path/to/final_model.pth'
    model, device = load_trained_faster_rcnn(weight_path)
    model.eval()  # set the model to eval mode
    # send model to devices
    model.to(device)
    # load the outlier detector
    filename = 'isolation_forest_model.sav'
    # load the model from disk
    outlier_model = pickle.load(open(filename, 'rb'))
    # a list to store all the last feature maps for all the samples in the training dataloader
    feature_maps = []
    # register the forward hook
    model.backbone.body.layer4[2].conv3.register_forward_hook(lambda module, input, output: feature_maps.append(output))
    # specify the cpu device
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    # set some counters to calculate the accuracy of the outlier detector
    count = 0  # number of input images containing the occluded pedestrians
    n_samples = 0  # number of occluded pedestrian instances
    n_tp = 0  # number of true positive predictions
    n_fn = 0  # number of false negative predictions

    # process the input image in training dataset one by one
    for idx, (image, target) in enumerate(tqdm.tqdm(test_loader)):
        # if no GT bboxes meet the requirements of the criteria choosing the GT, so skip
        if target[0]['No_bboxes_matched']:
            continue
        # send the input image to GPU
        image = [img.to(device) for img in image]
        with torch.no_grad():
            # get the width and height of the input image
            c_img, h_img, w_img = image[0].size()  # [C, H, W]
            # feed the input image to model
            outputs = model(image)
            # move the results to cpu
            outputs_cpu = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            count += 1
            # get the corresponding feature map regarding the particular input image
            feature_map = feature_maps.pop()[0].clone()
            # get the width and height of the feature map
            c_f, h_f, w_f = feature_map.shape  # with shape [C, H, W]
            # transform factors between bboxes and feature map
            w_factor, h_factor = w_f / w_img, h_f / h_img
            # apply the transform factor to the prediction bboxes
            refactored_bboxes = target[0]['boxes'].clone()
            refactored_bboxes[:, [0, 2]] *= w_factor
            refactored_bboxes[:, [1, 3]] *= h_factor
            # turn the float coords into the integer coords
            refactored_bboxes[:, [0, 1]] = torch.floor(refactored_bboxes[:, [0, 1]])  # x1, x2 are floored
            refactored_bboxes[:, [2, 3]] = torch.ceil(refactored_bboxes[:, [2, 3]])  # x2, y2 are ceiled
            # extract the activation within the areas of refactored bboxes
            extracted_feature_maps = []
            for refactored_bbox in refactored_bboxes.int():
                # extract the corresponding pieces of feature maps
                extracted_feature_map = feature_map[:, refactored_bbox[1]:min(refactored_bbox[3], h_f),
                                        refactored_bbox[0]:min(refactored_bbox[2], w_f)]  # shape [C, H_e, W_e]
                resized_tensor = torchvision.transforms.Resize(size=(1, 2))(
                    extracted_feature_map).cpu().numpy()
                # append the extracted activation
                extracted_feature_maps.append(resized_tensor)
            # reshape all the extraction to a feature matrix
            extraction_resized = np.array(extracted_feature_maps).reshape(len(extracted_feature_maps), -1)
            # pass the feature matrix to the outlier detector
            outlier_results = outlier_model.predict(extraction_resized)
            # calculate the number of samples, number of true positive and number of false negative
            n_samples += len(outlier_results)
            n_tp += outlier_results[outlier_results == 1].size
            n_fn += outlier_results[outlier_results == -1].size
            # plot several samples
            if count in range(0, 60, 3):
                occluded_plot(image, target, feature_map, refactored_bboxes, outlier_model, outlier_results)

    # calculate the accuracy rate of the outlier detector
    print(f'accuracy rate test: {n_tp}/{n_samples} = {round(n_tp / n_samples * 100, 3)}%')


if __name__ == '__main__':
    main()