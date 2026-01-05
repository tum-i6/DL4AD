import torch.nn as nn
from torchvision.models.detection import SSD
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.resnet import resnet50


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

        conv4_block1 = self.feature_extractor[-1][0]

        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)

    def forward(self, x):
        x = self.feature_extractor(x)
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
        steps=[8, 16, 32, 64, 100, 300],
    )
    model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)  # Loads the SSD300 detection model
    for param in model.parameters():
        param.requires_grad = True  # Sets all model weights to be trainable
    return model
