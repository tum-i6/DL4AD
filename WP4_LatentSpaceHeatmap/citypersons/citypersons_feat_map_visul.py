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
import argparse
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


def visualize_new(folder_name, img_path, activations, img, output, target=None, cfg=None):
    # define saving path
    img_name = img_path.split("/")[-1]
    root_path = f'{folder_name}/citypersons_demovideo'

    # tranche = img_path.split("/")[-2]
    # tranche_path = f'{root_path}/{tranche}'
    # if not os.path.exists(tranche_path):
    #     os.makedirs(tranche_path)

    # input images
    img_255_tensor = img * 255
    img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
    img_255 = img_255_tensor.numpy().transpose(1, 2, 0)  # CHW to HWC

    # 1. input image with dets
    # get the det bboxes coordinates from the target
    det_bboxes = output['boxes'].clone()
    det_scores = output['scores'].clone()
    img_det = cv2.cvtColor(np.copy(img_255), cv2.COLOR_RGB2BGR)  # RGB to BGR
    for bbox, conf in zip(det_bboxes, det_scores):
        label = f'{conf:.2f}'
        img_det = plot_one_box(bbox, img_det, color=[0, 47, 167], label=label, line_thickness=1)

    # det_path = f'{tranche_path}/detections'
    # if not os.path.exists(det_path):
    #     os.makedirs(det_path)

    # cv2.imwrite(f'{det_path}/{img_name}', img_det)

    # 2. heatmap
    original_h, original_w = img.size()[-2:]
    original_size = [original_h, original_w]
    i_h, i_w = tuple(original_size)  # size of input image
    for k, v in activations.items():
        if k != '0':
            continue

        # # channel-wise average the activations
        heatmap_avg = torch.mean(v, 1).squeeze(0).numpy()
        heatmap = (heatmap_avg - np.min(heatmap_avg)) / (np.max(heatmap_avg) - np.min(heatmap_avg))  # (H, W) 0-1

        #heatmap = 1 - heatmap  # TODO: put it into a function

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (i_w, i_h))

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # overlaying heatmap and resized input image.
        feat_overlay = (heatmap * 0.5 + img_255 * 0.5)  # RGB
        feat_overlay = cv2.cvtColor(feat_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR

        # 3. mask
        # plot the heatmap mask
        heatmap_mask = heatmap_avg
        heatmap_mask = cv2.resize(heatmap_mask, (i_w, i_h))
        heatmap_mask = 1 / (1 + np.exp(-heatmap_mask)) > .5  # binarization

        copy = np.copy(img_255)
        copy[heatmap_mask == 0] = 255

        mask_overlay = (0.75 * copy + 0.25 * img_255)
        mask_overlay = cv2.cvtColor(mask_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR

        for bbox, conf in zip(det_bboxes, det_scores):
            label = f'{conf:.2f}'
            mask_overlay = plot_one_box(bbox, mask_overlay, color=[0, 47, 167], label=label, line_thickness=1)

    return img_det, feat_overlay, mask_overlay

def visualize_main(folder_name, img_path, activations, img, output, target=None, cfg=None):
    # define saving path
    img_name = img_path.split("/")[-1]
    root_path = f'{folder_name}/citypersons_demovideo'

    # tranche = img_path.split("/")[-2]
    # tranche_path = f'{root_path}/{tranche}'
    # if not os.path.exists(tranche_path):
    #     os.makedirs(tranche_path)

    # input images
    img_255_tensor = img * 255
    img_255_tensor = img_255_tensor.to(torch.device("cpu"), torch.uint8)
    img_255 = img_255_tensor.numpy().transpose(1, 2, 0)  # CHW to HWC

    # 1. input image with dets
    # get the det bboxes coordinates from the target
    det_bboxes = output['boxes'].clone()
    det_scores = output['scores'].clone()
    img_det = cv2.cvtColor(np.copy(img_255), cv2.COLOR_RGB2BGR)  # RGB to BGR
    for bbox, conf in zip(det_bboxes, det_scores):
        label = f'{conf:.2f}'
        img_det = plot_one_box(bbox, img_det, color=[0, 47, 167], label=label, line_thickness=1)

    # det_path = f'{tranche_path}/detections'
    # if not os.path.exists(det_path):
    #     os.makedirs(det_path)

    # cv2.imwrite(f'{det_path}/{img_name}', img_det)

    # 2. heatmap
    original_h, original_w = img.size()[-2:]
    original_size = [original_h, original_w]
    i_h, i_w = tuple(original_size)  # size of input image

    if cfg['model_name'] != 'SSD300-ResNet50-OD':
        heatmap = torch.mean(activations['0'], 1).squeeze(0).numpy()
    else:
        heatmap = torch.mean(activations, 1).squeeze(0).numpy()
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # (H, W) 0-1
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (i_w, i_h))

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # overlaying heatmap and resized input image.
    feat_overlay = (heatmap * 0.5 + img_255 * 0.5)  # RGB
    feat_overlay = cv2.cvtColor(feat_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR

    # 3. mask
    # plot the heatmap mask
    heatmap_mask = heatmap
    heatmap_mask = cv2.resize(heatmap_mask, (i_w, i_h))
    heatmap_mask = 1 / (1 + np.exp(-heatmap_mask)) > .5  # binarization

    copy = np.copy(img_255)
    copy[heatmap_mask == 0] = 255

    mask_overlay = (0.75 * copy + 0.25 * img_255)
    mask_overlay = cv2.cvtColor(mask_overlay.astype('uint8'), cv2.COLOR_RGB2BGR)  # BGR

    for bbox, conf in zip(det_bboxes, det_scores):
        label = f'{conf:.2f}'
        mask_overlay = plot_one_box(bbox, mask_overlay, color=[0, 47, 167], label=label, line_thickness=1)

    return img_det, feat_overlay, mask_overlay
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
def inference_video(model, device, folder_name, feature_maps, tag, img_list, conf_thre=0.5):
    cpu_device = torch.device("cpu")
    model = model.eval()

    # video maker!
    height, width, layers = cv2.imread(img_list[0]).shape

    # choose codec according to format needed
    root_path = f'{folder_name}/{tag}_demovideo'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    det_video = cv2.VideoWriter(f'{root_path}/{tag}_detection_conf_{conf_thre}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                (width, height))
    feat_video = cv2.VideoWriter(f'{root_path}/{tag}_heatmap_conf_{conf_thre}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                 (width, height))
    mask_video = cv2.VideoWriter(f'{root_path}/{tag}_mask_conf_{conf_thre}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10,
                                 (width, height))

    for idx, img in tqdm.tqdm(enumerate(img_list)):
        # if idx > 800:
        #     exit()
        img_file_name=img.split('/')[-1].split('.')[0]
        img_folder_name=img.split('/')[-2]
        # infer the input image
        image = img2tensor(img, device)
        # get the results
        results = model([image])
        results = [{k: v.to(cpu_device) for k, v in t.items()} for t in results]
        results = [{k: v[t['scores'] >= conf_thre] for k, v in t.items()} for t in results]

        # get the feature maps from the model
        activations = feature_maps.pop()
        activations = {k: v.detach().cpu() for k, v in activations.items()}

        img_det, heatmap, mask = visualize_new(folder_name, img, activations, image, results[0])
        det_video.write(img_det)
        feat_video.write(heatmap)
        mask_video.write(mask)

    feat_video.release()
    det_video.release()
    mask_video.release()

@torch.inference_mode()
def inference_image(model, device, folder_name, feature_maps, tag, cfg, img_list, conf_thre=0.5):
    cpu_device = torch.device("cpu")
    model = model.eval()

    # video maker!
    height, width, layers = cv2.imread(img_list[0]).shape

    # choose codec according to format needed
    root_path = f'{folder_name}/{tag}_demovideo'
    if not os.path.exists(root_path):
        os.makedirs(root_path)



    for idx, img in tqdm.tqdm(enumerate(img_list), total=len(img_list)):
        # if idx > 800:
        #     exit()
        img_file_name=img.split('/')[-1].split('.')[0]
        img_folder_name=img.split('/')[-2]
        # infer the input image
        image = img2tensor(img, device)
        # get the results
        results = model([image])
        results = [{k: v.to(cpu_device) for k, v in t.items()} for t in results]
        results = [{k: v[t['scores'] >= conf_thre] for k, v in t.items()} for t in results]

        # get the feature maps from the model
        activations = feature_maps.pop()
        if cfg['model_name'] != 'SSD300-ResNet50-OD':
            activations = {k: v.detach().cpu() for k, v in activations.items() if k == '0'}
        else:
            activations = activations.detach().cpu()

        img_det, heatmap, mask = visualize_main(folder_name, img, activations, image, results[0], cfg=cfg)
        #print(f'{root_path}/{img_folder_name}/{img_file_name}_det.png')
        cv2.imwrite(f'{root_path}/{img_file_name}_det.png', img_det)
        cv2.imwrite(f'{root_path}/{img_file_name}_heatmap.png', heatmap)
        cv2.imwrite(f'{root_path}/{img_file_name}_mask.png', mask)


def path_list(data_path):
    total_img_list = []

    for subset in os.listdir(data_path):
        #if subset=='stuttgart_01':
        img_path = f'{data_path}/{subset}'
        img_list = glob.glob(f'{img_path}/*.png')
        total_img_list += img_list
        img_list = glob.glob(f'{img_path}/*.jpg')
        total_img_list += img_list

    return sorted(total_img_list)


def main():
    # read the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='.config/eval_config.yaml',
                        help='Path to the config yaml file, which contains the train parameters.')
    args = parser.parse_args()
    with open(args.config) as yaml_file:
        cfg = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # Loads the detection model and the device
    model, device = load_model(cfg)

    # plot curves and save a couple of images
    saving_path = '/'.join(cfg['pretrained_weights_path'].split('/')[:-1])

    print(saving_path)
    contra_weighting_factor = int(saving_path.split('/')[-1].split('_')[-1])  # in percentage
    saving_path = 'path/to/output_dir'
    model_name = cfg['model_name'].split('-')[0]  # e.g. FCOS
    dataset_name = cfg['dataset']  # e.g. CityPersons

    tag = f'{model_name}_wf_{contra_weighting_factor}'  # e.g. CityPersons_FCOS_wf_10
    print(tag)

    # register hooks on the feature maps
    feature_maps = []
    model.backbone.register_forward_hook(lambda module, input, output: feature_maps.append(output))

    root_path = '/path/to/examples_dir'
    #img_path_list = path_list(root_path)
    img_path_list = glob.glob(f'{root_path}/*.png')
    img_path_list += glob.glob(f'{root_path}/*.jpg')

    #inference_video(model, device, saving_path, feature_maps, tag=tag, img_list=img_path_list, conf_thre=0.25)
    inference_image(model, device, saving_path, feature_maps, tag=tag, img_list=img_path_list, cfg=cfg, conf_thre=0.25)


if __name__ == '__main__':
    main()
