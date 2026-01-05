import glob

from torchvision import transforms, models
import numpy as np
import torch.nn as nn
import torch
import argparse
import cv2
import os
from PIL import Image
import pandas

cnns = ['VGG16', 'VGG19', 'InceptionV3', 'ResNet18', 'ResNet50', 'ResNet101', 'ResNet152', 'WideResNet50',
        'WideResNet101', 'ResNeXt50', 'ResNeXt101']

weights = [
    'path/to/model1/weights.pt',
    'path/to/model2/weights.pt',
]

frames_dir = '/path/to/frames_dir'
df_output_path='//path/to/output_dir'
model_name = cnns[0]
model_weights = weights[0]
target_attack = False
true_target_class = 1
attack_target_class = 5





# Model zoo (all supported models)
models = {'VGG16': models.vgg16_bn(),
          'VGG19': models.vgg19_bn(),
          'InceptionV3': models.inception_v3(),
          'ResNet18': models.resnet18(),
          'ResNet50': models.resnet50(),
          'ResNet101': models.resnet101(),
          'ResNet152': models.resnet152(),
          'WideResNet50': models.wide_resnet50_2(),
          'WideResNet101': models.wide_resnet101_2(),
          'ResNeXt50': models.resnext50_32x4d(),
          'ResNeXt101': models.resnext101_32x8d()}


def load_model(model_name, device_ids, weights_path):
    """
    Loads the classification model and the device.

    :param model_name: Name of the classification model, which should be loaded.
    :param device_ids: Device IDs for the GPUs, which should be used for training/evaluation.
    :param weights_path: Path to the model weights.
    :return: The classification model and the device, which are being used for training/evaluation.
    """
    # Sets the device for training
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\nDevice used for training/evaluation: {device}')
    print(f'\nLoading {model_name} model (and weights).')

    # Loads the selected classification model and sets the amount of output parameters to 43 (amount of GTSRB classes)
    model = models[model_name]  # Instantiates the model

    # Sets the amount of output parameters for the VGG models
    if 'VGG' in model_name:
        model.classifier[6] = nn.Linear(4096, 43)
    # Sets the amount of output parameters for the InceptionV3 model
    elif model_name == 'InceptionV3':
        model.AuxLogits.fc = nn.Linear(768, 43)
        model.fc = nn.Linear(2048, 43)
    # Sets the amount of output parameters for the ResNet models (also includes ResNeXt) and adds a dropout layer
    elif 'Res' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_ftrs, 43)
        )

    model = nn.DataParallel(model)  # Sets the model to work in parallel on all available GPUs

    model.load_state_dict(torch.load(weights_path, map_location=device))  # Loads the model weights
    model = model.to(device)  # Moves the model to the device
    print(model_name + ' has been loaded!\n')

    return model, device


# Loads the classification model
model, device = load_model(model_name, '0', model_weights)
model.eval()

frames = sorted(os.listdir(frames_dir))

preprocess = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Resize((32, 32)),  # ToDo
        #transforms.Resize((299, 299)),
        transforms.Normalize([0.33999264, 0.31174879, 0.3209361], [0.27170149, 0.2598821, 0.26575037]),
    ])

correct_preds = 0
wrong_preds = 0
attack_target_preds = 0
k = 0
df = pandas.DataFrame(columns=[
            'frame_id',
            'label',
            'img',
            'pred'
        ])
for frame_file in sorted(glob.glob(frames_dir+'/*.png')):
    print(f"Progress: {k/len(frames) * 100 :.2f}%")
    img = Image.open(frame_file).convert('RGB')
    frame_id=frame_file.split('.')[0].split('_')[-1]
    prep_img = preprocess(img).unsqueeze(0).to(device)

    output = model(prep_img)
    out_soft=torch.softmax(output,dim=1)

    if torch.argmax(output, 1) == true_target_class:
        correct_preds += 1
    else:
        wrong_preds += 1

    if torch.argmax(output, 1) == attack_target_class:
        attack_target_preds += 1
    df.loc[-1] = [frame_id,true_target_class,img,out_soft.cpu().detach().numpy()]  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()
    k += 1
df=df.sort_values(by=['frame_id'],ascending=True,ignore_index=True)
df.to_pickle(df_output_path+'.p')
print(model_name)
print(frames_dir)
print(f'Victim Class: {true_target_class}')
print(f'Attack Target Class: {attack_target_class}')
print(f'Prediction Accuracy: {correct_preds / len(frames) * 100}')
print(f'Missclassifications: {wrong_preds / len(frames) * 100}')
print(f'Attack Target Prediction Accuracy: {attack_target_preds / len(frames) * 100}')






