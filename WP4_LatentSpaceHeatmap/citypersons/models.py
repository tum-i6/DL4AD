import os
import torch
import torch.nn as nn
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.resnet import resnet50

from retinanet_torchvision.retinanet import retinanet_resnet50_fpn
from faster_rcnn_torchvision.faster_rcnn import fasterrcnn_resnet50_fpn
from faster_rcnn_torchvision.mask_rcnn import maskrcnn_resnet50_fpn
from fcos_torchvision.fcos import fcos_resnet50_fpn
from ssd_torchvision.ssd import SSD

# a dict to store the default confidence threshold applied by the listed models
standard_conf_thres = {
    'FasterRCNN-ResNet50-OD': .05,

    'MaskRCNN-ResNet50-IS': .05,

    'FCOS-ResNet50-OD': .2,

    'RetinaNet-ResNet50-OD': .05,

    'SSD300-ResNet50-OD': .01,
}


class ResNet(nn.Module):
    """
    Defines the ResNet class for the ResNet50 backbone network. Note that this implementation was taken from:
    https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/
    """

    def __init__(self, pretrained_backbone=True):
        super().__init__()
        backbone = resnet50(pretrained_backbone)
        self.out_channels = [1024, 512, 512, 256, 256, 256]

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:7])

        conv4_block1 = self.feature_extractor[-1][0]  # conv->sequential; block -> bottleneck
        # (6): Sequential(
        #     (0): Bottleneck(

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)  # shape torch.Size([1, 1024, 38, 38])
        return x


def ssd300_resnet50(pretrained_backbone, num_classes, **kwargs):
    """
    This function loads a SSD300 detection model with a custom ResNet50 backbone network.

    :param pretrained_backbone: Whether to initialize the pretrained backbone weights, which were optimized on ImageNet.
    Options: True or False.
    :param num_classes: The amount of classes, which should be detected. Note that num_classes must be at least 2 to
    account for the background class. The value must be of type integer.
    :return: A PyTorch SSD300 detection model.
    """
    backbone = ResNet(pretrained_backbone)  # Loads the backbone network
    # Defines the detection anchors
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300],  # stride
    )
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)  # Loads the SSD300 detection model
    for param in model.parameters():
        param.requires_grad = True  # Sets all model weights to be trainable
    return model


# a dict to store the default models
models = {
    'FasterRCNN-ResNet50-OD': fasterrcnn_resnet50_fpn(
        pretrained_backbone=True,
        trainable_backbone_layers=5,
        num_classes=2,
    ),

    'MaskRCNN-ResNet50-IS': maskrcnn_resnet50_fpn(
        pretrained_backbone=True,
        trainable_backbone_layers=5,
        num_classes=2,
    ),

    'FCOS-ResNet50-OD':
        fcos_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2,
        ),

    'RetinaNet-ResNet50-OD':
        retinanet_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2,
        ),

    'SSD300-ResNet50-OD':
        ssd300_resnet50(
            pretrained_backbone=True,
            num_classes=2
        ),

}

# a dict storing all the models without conf_thres for the modified post-processing algorithm
models_no_conf_thres = {
    'FasterRCNN-ResNet50-OD': fasterrcnn_resnet50_fpn(
        pretrained_backbone=True,
        trainable_backbone_layers=5,
        num_classes=2,
        # output conf threshold
        box_score_thresh=0.0,
    ),

    'MaskRCNN-ResNet50-IS': maskrcnn_resnet50_fpn(
        pretrained_backbone=True,
        trainable_backbone_layers=5,
        num_classes=2,
        # output conf threshold
        box_score_thresh=0.0,
    ),

    'FCOS-ResNet50-OD':
        fcos_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2,
            # output score threshold
            score_thresh=0.0
        ),

    'RetinaNet-ResNet50-OD':
        retinanet_resnet50_fpn(
            pretrained_backbone=True,
            trainable_backbone_layers=5,
            num_classes=2,
            # output score threshold
            score_thresh=0.0,
        ),

    'SSD300-ResNet50-OD':
        ssd300_resnet50(
            pretrained_backbone=True,
            num_classes=2,
            # output score threshold
            score_thresh=0.0,
        ),

}


def load_model(cfg, zero_conf_thres=False):
    """
    Loads the detection model and the device, used for training/evaluation. Note that only one GPU can be utilized at
    the time.

    :param cfg: A configuration dictionary containing the name of the detection model, the GPU ID and optionally a
    string path to a model weights file, which should be used for fine-tuning.
    :param zero_conf_thres: bool input, true for loading the model without any confidence threshold
    :return: The detection model and the device, which are being used for training/evaluation.
    """
    # Sets the GPU, which should be used for training/evaluation
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_id'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'\nDevice used for training/evaluation: {device}')
    print(f"\nLoading {cfg['model_name']} model.")

    # Loads the selected detection model
    if zero_conf_thres:
        print('Note: 0-confidence-threshold-model instantiated! ')
        model = models_no_conf_thres[cfg['model_name']]  # Instantiates the model
    else:
        model = models[cfg['model_name']]  # Instantiates the model
    model.to(device)  # Moves the model to the device

    # Loads the pretrained model weights, if specified
    if 'pretrained_weights_path' in cfg:
        if cfg['pretrained_weights_path']:
            if os.path.exists(cfg['pretrained_weights_path']):
                checkpoint = torch.load(cfg['pretrained_weights_path'])
                model.load_state_dict(checkpoint['model'])
                print(f'pre-trained model has been found and loaded!')
            else:
                print(f'pre-trained model cannot be found, training from scratch ...')
        else:
            print(f'pre-trained model cannot be found, training from scratch ...')
    else:
        print('not specific pre-trained weights, training from scratch ...')
    return model, device
