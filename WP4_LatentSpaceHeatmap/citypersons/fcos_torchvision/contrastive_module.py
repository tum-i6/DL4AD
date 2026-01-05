import torch
from torch import nn, Tensor

from typing import Optional, List, Dict, Tuple
from torch.nn import BCEWithLogitsLoss
import torchvision.transforms as T
import torchvision.transforms.functional as F


def heatmap_avg_channels(features_list: List[Tensor]):
    """
    average the feature maps across the channel axis.
    :param features_list: a list containing the extracted feature maps
    :return: a list containing the heatmap
    """
    heatmap_list = []
    for layer, feat in enumerate(features_list):  # for every RPN layer
        heatmap = torch.mean(feat, 1)  # [B, C, H, W] -> [B, H, W]

        # --------------test block begins--------------
        if torch.isinf(heatmap).sum().item() != 0:
            print(f'Inf values found at heatmap layer {layer + 1}: {torch.isinf(heatmap).sum().item()}')

        if torch.isnan(heatmap).sum().item() != 0:
            print(f'nan values found at heatmap {layer + 1}: {torch.isnan(heatmap).sum().item()}')
        # --------------test block ends--------------

        heatmap_list.append(heatmap)

    return heatmap_list  # List[Tensor]


def coarse_binary_mask(boxes: List[Tensor],
                       image_shapes: List[Tuple[int, int]]
                       ):
    """
    make the gt binary mask for each input image.
    :param boxes: the ground truth bounding boxes list
    :param image_shapes: the input images shape list.
    :return: a list containing the gt binary mask.
    """
    device = boxes[0].device
    mask_list = []  # List[Tensor]

    for b, bbox in enumerate(boxes):
        # label smoothing for neg. pixels: 0.1 instead of 0
        mask = torch.ones(image_shapes[b], device=device, dtype=torch.float) * 0.1
        for gt in bbox:
            x1, y1, x2, y2 = tuple(gt.to(torch.long))
            # label smoothing for pos. pixels: 0.9 instead of 1
            mask[y1: y2, x1: x2] = .9

        mask_list.append(mask)

    return mask_list  # List[Tensor]


def resize_mask(mask_list: List[Tensor],
                heatmap_list: List[Tensor],
                ):
    """
    down-size the gt binary mask to the same size of the heatmaps in the latent space.
    :param mask_list: a list containing all the gt binary mask
    :param heatmap_list: a list containing all the heatmaps.
    :return: a list containing all the resized gt binary mask
    """
    first_feat = heatmap_list[0]  # for the feature map with the smallest stride
    f_h, f_w = first_feat.shape[-2:]
    # resize begins!
    resized_mask_list = [
        T.Resize(
            size=(f_h, f_w),
            interpolation=F.InterpolationMode.NEAREST)(x.unsqueeze(0)) for x in mask_list  # expected size [B, C, ...]
    ]

    # --------------test block begins--------------
    for idx, x in enumerate(resized_mask_list):  # for every gt mask
        if torch.isinf(x).sum().item() != 0:
            print(f'Inf values found at resized gt mask {idx + 1}: {torch.isinf(x).sum().item()}')

        if torch.isnan(x).sum().item() != 0:
            print(f'nan values found at resized gt mask {idx + 1}: {torch.isnan(x).sum().item()}')

        if len(torch.unique(x)) != 2 and len(torch.unique(x)) != 1:
            print(f'resized binary mask {idx + 1} has more than 2 values: {torch.unique(x)}')
    # --------------test block ends--------------

    resized_mask_list = [x.squeeze(0) for x in resized_mask_list]

    return resized_mask_list


def bce_loss(pred: Tensor,
             binary_label: Tensor,
             loss_func=BCEWithLogitsLoss(reduction='none'),  # w/ integrated Sigmoid
             ):
    """
    the function to calculate the new loss for one single input image during the forward pass.
    :param pred: the predicted heatmap from the latent space
    :param binary_label: the resized gt binary mask
    :param loss_func: the loss function
    :return: two loss terms namely pos and neg for pedestrian pixels and background pixels respectively.
    """
    # --------------test block begins--------------
    # [x] THIS ONE test is nan value in pred triggers this problem
    # - If the parameters show invalid values, most likely the gradients were too large [gradients exploding]
    # - the model was diverging, and the parameters were overflowing.
    # https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/3
    if torch.isnan(pred).sum().item() != 0:
        print(f'nan values in pred: {torch.isnan(pred).sum().item()}')
    # --------------test block ends--------------

    loss = loss_func(pred, binary_label)

    # --------------test block begins--------------
    # test if nan loss
    if torch.isnan(loss).sum().item() != 0:
        print(f'loss is nan!')
        print(f'pred: {torch.isnan(pred).sum().item()} max: {torch.max(pred)} ')
        print(f'label: {torch.isnan(binary_label).sum().item()} max: {torch.max(binary_label)}')
    # --------------test block ends--------------

    # positive pixels loss
    if len(torch.unique(binary_label)) == 1:  # only background
        loss_pos = torch.tensor(float('nan'))
    else:
        loss_pos = torch.nanmean(loss[binary_label == torch.unique(binary_label)[1]])

    # negative pixels loss
    loss_neg = torch.nanmean(loss[binary_label == torch.unique(binary_label)[0]])

    return loss_pos, loss_neg


def batch_loss(features: List[Tensor],
               mask_list: List[Tensor], ):
    """
    calculate the new loss over the whole batch.
    :param features: a list containing the heatmaps
    :param mask_list: a list containing the resized binary mask
    :return: two loss terms namely pos and neg for pedestrian pixels and background pixels respectively.
    """
    loss_pos, loss_neg = 0, 0
    # counting the number of non-empty image in the batch
    n_batch_pos = 0
    # only the feature maps of the 1st layer are needed
    first_features = features[0]
    for b, mask in enumerate(mask_list):
        # get the losses for each input image, p for positive and n for negative
        p_loss, n_loss = bce_loss(first_features[b], mask)

        if not torch.isnan(p_loss):  # the current image is not empty (gt inside)
            loss_pos = loss_pos + p_loss
            n_batch_pos = 1 + n_batch_pos

        loss_neg = n_loss + loss_neg

    if n_batch_pos != 0:
        loss_pos = loss_pos / n_batch_pos
    else:  # all the image in the batch are empty (no gt inside)
        loss_pos = torch.tensor(float('nan'))
    # average the loss over the batch size
    loss_neg = loss_neg / len(mask_list)

    return loss_pos, loss_neg


class ContrastiveLoss(nn.Module):
    """
    a class to add the new contrastive loss during the training phase.
    """

    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        # here only the feature maps of the 1st layer (index 0) are needed
        self.featmap_names = ['0']

    def forward(self,
                features,  # type: Dict[str, Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None  # type: Optional[List[Dict[str, Tensor]]]
                ):
        """
        the key forward function to calculate the contrastive loss.
        :param features: the feature (pyramid) maps collected during the forward pass.
        :param image_shapes: the shapes of the actually input imaged
        :param targets: a list containing the ground truth. Optional, needed by training
        :return:
        """
        # placeholder to save the needed feature maps
        x_filtered = []
        for k, v in features.items():
            if k in self.featmap_names:
                x_filtered.append(v)

        device = features['0'].device
        # ground truth bboxes
        boxes = [target['boxes'].to(device) for target in targets]  # List[Tensor[N, 4]]
        # make the ground truth binary mask for each input image in the image
        mask_list = coarse_binary_mask(boxes, image_shapes)  # for every img at the batch
        # process the extracted feature maps by averaging across channel axis
        heatmap_list = heatmap_avg_channels(x_filtered)  # for every level of FPN
        # downsize the gt binary mask to the same as the heatmaps
        resize_mask_list = resize_mask(mask_list, heatmap_list)
        # calculate the loss, pos for pedestrian pixels, neg for background pixels
        loss_pos, loss_neg = batch_loss(heatmap_list, resize_mask_list)

        assert not torch.isnan(loss_neg)  # test if the neg loss is nan

        if torch.isnan(loss_pos):  # without any gt objects in the whole batch!
            loss = loss_neg  # only return the loss of background
        else:
            loss = (loss_pos + loss_neg) / 2  # take the mean to avid the class imbalance problem

        return {'contrastive_loss': loss}
