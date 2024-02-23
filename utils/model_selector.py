import torch
import torchvision
from torch import nn


def model_selector(model, number_of_classes, additional_text=None):

    if additional_text:
        model_name = f'{model.__class__.__name__}_{additional_text}'
    else:
        model_name = model.__class__.__name__

    if model_name == 'ResNet':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, number_of_classes)

    elif model_name == 'MobileNetV3':
        model.classifier[-1] = nn.Linear(1280, number_of_classes)

    elif model_name == 'MobileNetV2':
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, number_of_classes)

    elif model_name == 'EfficientNet':
        model.classifier[1] = nn.Linear(1280, number_of_classes)

    else:
        print(f'[ERROR]: {model_name} is not a valid model name. Please check the model initialization!')
