import os
import datetime
import json
import numpy as np
import glob
import cv2
import random
import yaml
from matplotlib import pyplot as plt, patches, patheffects
from matplotlib.pyplot import figure
from scipy.io import loadmat
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.transforms.functional import convert_image_dtype
import torch
from torchvision.utils import draw_segmentation_masks
import tqdm

from models import load_model

category_dict = {
    'Motorcycle': 0,
    'Pedestrian': 1,
    'Bicycle': 2,
    'Bus': 3,
    'Van': 4,
    'Trailer': 5,
    'Truck': 6,
    'Car': 7
}

occ_dict = {'PARTIALLY_OCCLUDED': 1, 'NOT_OCCLUDED': 0, 'MOSTLY_OCCLUDED': 2}


def img2tensor(path, device):
    # append the detection results for a single input image
    img = read_image(path)
    img = convert_image_dtype(img, dtype=torch.float).to(device)  # int -> float
    return img


def visualize(folder_name, img_path, activations, img, target, output):
    img_name = img_path.split("/")[-1]

    # saving path
    img_gts_rot_path = img_path.replace('images', 'images_with_gts')
    img_gts_rot_path = "/".join(img_gts_rot_path.split("/")[:-1])
    if not os.path.exists(img_gts_rot_path):
        os.makedirs(img_gts_rot_path)

    feat_maps_root_path = img_path.replace('images', f'feat_maps/{folder_name}')
    feat_maps_root_path = "/".join(feat_maps_root_path.split("/")[:-1])
    if not os.path.exists(feat_maps_root_path):
        os.makedirs(feat_maps_root_path)

    mask_root_path = img_path.replace('images', f'masks/{folder_name}')
    mask_root_path = "/".join(mask_root_path.split("/")[:-1])
    if not os.path.exists(mask_root_path):
        os.makedirs(mask_root_path)

    # size of input image
    original_h, original_w = img.size()[-2:]
    original_size = [original_h, original_w]
    i_h, i_w = tuple(original_size)

    # input images
    img_255_tensor = img * 255
    img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
    img_255 = img_255_tensor.numpy().transpose(1, 2, 0)

    # detection results
    # get the GT bboxes coordinates from the target
    tar_bboxes = target['boxes'].clone().cpu()
    if len(tar_bboxes) != 0:
        tar_bboxes[:, [2]] = tar_bboxes[:, [2]] - tar_bboxes[:, [0]]
        tar_bboxes[:, [3]] = tar_bboxes[:, [3]] - tar_bboxes[:, [1]]

    # get the det bboxes coordinates from the target
    det_bboxes = output['boxes'].clone()
    if len(det_bboxes) != 0:
        det_bboxes[:, [2]] = det_bboxes[:, [2]] - det_bboxes[:, [0]]
        det_bboxes[:, [3]] = det_bboxes[:, [3]] - det_bboxes[:, [1]]
        det_scores = output['scores'].clone()

    # Plot input images and Gt annotations
    dpi = 300
    # fig = plt.figure()
    # ax2 = fig.add_subplot()
    # plt.imshow(img_255)
    #
    # # add boxes
    # for idx, box in enumerate(tar_bboxes):
    #     rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='#88f5c5',
    #                              facecolor='none')
    #     # ax.add_patch(rect)l
    #     # ax2.set_title('Input image with Ground Truth')
    #     ax2.add_patch(rect)
    #
    # plt.axis('off')
    # plt.tight_layout()
    # # fig.savefig(f'{img_gts_rot_path}/{img_name}', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
    # plt.show()  # input image with Gts

    # plt.cla()
    # plt.close(fig)

    for k, v in activations.items():
        if k != '0':
            continue
        # print(f'layer {k} with feature map shape {v.shape}')

        # # channel-wise average the activations
        heatmap_avg = torch.mean(v, 1).squeeze(0).numpy()
        # heatmap from layers with (H, W) 0-1
        heatmap = (heatmap_avg - np.min(heatmap_avg)) / (np.max(heatmap_avg) - np.min(heatmap_avg))  # (H, W) 0-1

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (i_w, i_h))

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        b, g, r = cv2.split(heatmap)
        heatmap = cv2.merge([r, g, b])

        # overlaying heatmap and resized input image.
        result = (heatmap * 0.5 + img_255 * 0.5) / 255

        # plot the overlaying
        plt.imshow(result)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{feat_maps_root_path}/{img_name}', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        # plt.show()

        plt.cla()

        # # plot the heatmap mask
        # heatmap_mask = heatmap_avg
        # heatmap_mask = cv2.resize(heatmap_mask, (i_w, i_h))
        # heatmap_mask = heatmap_mask > 0
        #
        # copy = np.copy(img_255)  # np.copy(img_255)
        # copy[heatmap_mask == 0] = 255
        #
        # overlay = (0.75 * copy + 0.25 * img_255) / 255
        #
        # # plot the overlaying
        # plt.imshow(overlay)
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig(f'{mask_root_path}/{img_name}', dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        # # plt.show()
        #
        # plt.cla()

        # # Plot the detection results
        # fig = plt.figure()
        # ax2 = fig.add_subplot()
        # plt.imshow(img_255)
        #
        # # instance segmentation
        # if 'masks' in output:
        #     proba_threshold = 0.5
        #     bool_mask = output['masks'] > proba_threshold
        #     bool_mask = bool_mask.squeeze(1)
        #     img_mask = draw_segmentation_masks(img_255_tensor, bool_mask, alpha=0.9)
        #     plt.imshow(np.asarray(F.to_pil_image(img_mask.detach())))
        #
        # plt.axis('off')
        # # add boxes
        # for idx, (box, score) in enumerate(zip(det_bboxes, det_scores)):
        #     rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='b',
        #                              facecolor='none')
        #     # ax.add_patch(rect)
        #     ax2.set_title('Input image with Predictions')
        #     ax2.add_patch(rect).set_path_effects([patheffects.Stroke(linewidth=0.5, foreground='black'),
        #                                           patheffects.Normal()])
        #     ax2.text(box[0], (box[1] - 50), str(round(100 * score.item(), 1)), verticalalignment='top',
        #              color='white', fontsize=6.5, weight='bold').set_path_effects([patheffects.Stroke(
        #         linewidth=3, foreground='black'), patheffects.Normal()])
        #
        # plt.tight_layout()
        # plt.show()


def visualize_new(folder_name, img_path, activations, img, target, output):
    # input images
    img_255_tensor = img * 255
    img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
    img_255 = img_255_tensor.numpy().transpose(1, 2, 0)

    # detection results
    # get the GT bboxes coordinates from the target
    tar_bboxes = target['boxes'].clone().cpu()
    img_tar = np.copy(img_255)
    for bbox in tar_bboxes:
        img_tar = plot_one_box(bbox, img_tar, color=[129, 216, 208])

    figure(dpi=100)
    plt.imshow(img_tar)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # get the det bboxes coordinates from the target
    det_bboxes = output['boxes'].clone()
    det_scores = output['scores'].clone()
    img_det = np.copy(img_255)
    for bbox, conf in zip(det_bboxes, det_scores):
        label = f'pedestrian {conf:.2f}'
        img_det = plot_one_box(bbox, img_det, color=[0, 47, 167], label=label)

    figure(dpi=100)
    plt.imshow(img_det)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def cuboid2box(cuboid):
    x_list = []
    y_list = []

    for p in cuboid.keys():
        x_list.append(float(cuboid[p]['x']))
        y_list.append(float(cuboid[p]['y']))
    assert len(x_list) == len(y_list)

    # convert 8 points 3d bboxes coordinates to 4 points 2d bboxes
    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    box = np.array([x_min, y_min, x_max, y_max], dtype=int)

    return box


@torch.inference_mode()
def inference(model, device, folder_name, feature_maps, img_list, ann_list):
    cpu_device = torch.device("cpu")
    model = model.eval()
    n_pedestrians = 0

    for idx, (img, ann) in tqdm.tqdm(enumerate(zip(img_list, ann_list))):
        if idx > 3:
            exit()
        annotation = json.load(open(ann))
        labels = annotation['labels']

        category_list = []
        bbox_list = []
        occ_list = []

        for label in labels:
            category = str(label['category'])
            if category not in ["Pedestrian", "Motorcycle", "Bicycle"]:
                continue
            category_list.append(category_dict[category])
            attributes = label['attributes']
            occ_list.append(occ_dict[attributes['Occluded']['value']])
            cuboid = label['cuboid']['points']
            bbox = cuboid2box(cuboid)
            bbox_list.append(bbox)
            n_pedestrians += 1

        target = [
            {
                'boxes': torch.as_tensor(np.array(bbox_list), dtype=torch.float32),
                'labels': torch.as_tensor(category_list, dtype=torch.int64),
                'occlusion': torch.as_tensor(occ_list, dtype=torch.int64),
            }
        ]

        # infer the input image
        image = img2tensor(img, device)
        # get the results
        results = model([image])
        results = [{k: v.to(cpu_device) for k, v in t.items()} for t in results]

        # get the feature maps from the model
        activations = feature_maps.pop()
        activations = {k: v.detach().cpu() for k, v in activations.items()}
        # for k, v in activations.items():
        #     print(f'FPN level {k}')
        #     print(f'shape {v.shape}')

        visualize_new(folder_name, img, activations, image, target[0], results[0])

    print(f'There are {n_pedestrians} pedestrian are found in in total {len(img_list)} images!')
    print(f'In average there are {round(n_pedestrians / len(img_list), 3)} pedestrians/ image')


def path_list(data_path):
    total_img_list = []
    total_ann_list = []

    for subset in os.listdir(data_path):
        img_path = f'{data_path}/{subset}/images'

        for phase in os.listdir(img_path):
            img_list = glob.glob(f'{img_path}/{phase}/*.png')
            ann_list = [img_name.replace('images', 'labels').replace('png', 'json') for img_name in img_list]

            total_img_list += img_list
            total_ann_list += ann_list
    assert len(total_img_list) == len(total_ann_list)

    return total_img_list, total_ann_list


def main():
    # read the config file
    cfg_path = '.config/eval_config.yaml'

    with open(cfg_path) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # infer the folder containing the model
    folder_name = cfg['pretrained_weights_path'].split('/')[-2]

    # register hooks on the feature maps
    feature_maps = []
    model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))

    root_path = '/data/ai_data_and_models/a9-dataset-crossing-tum'
    img_path_list, ann_path_list = path_list(root_path)

    inference(model, device, folder_name, feature_maps, img_list=img_path_list, ann_list=ann_path_list)


if __name__ == '__main__':
    main()
