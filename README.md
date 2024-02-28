FLOWER CLASSIFICATION WITH PYTORCH
===============================

This repository contains code for training and evaluating for flower classification . The code is organized into three main scripts:

1.  `train.py`: This script is responsible for training models on a custom flower classification dataset. It includes functions for loading the dataset, training the model, saving the trained weights, and plotting the training curves.

2.  `evaluate.py`: This script performs evaluating on the test dataset, it calculates different metrics like, test accuracy, precision, F1-score. It also creates classification reports and heatmap based on the confusion matrix. It also includes inference on 8 random images from the testing dataset.

3.  `dataset.py`: This module defines a custom dataset class for handling the dataset. It includes data loading, dataset splitting, augmentations.

Usage
-----

### 1\. Training the Model

To train the models (transfer learning), execute the following command:

```bash
python train.py
```
This will train ResNet18, EfficientNetB0, MoblieNetv3, MobilNetv2 models for a specified number of epochs using the custom dataset.

### 2\. Evaluate with the trained models

For inference using the trained models, run the following command:
```bash
python evalaute.py`
```
This script loads the trained model, calculates the metrics, plot the classification report and the confusion matrix heatmap.

Dataset
-------

The flower classification dataset used in this project can be downloaded from [here](https://www.kaggle.com/datasets/shahidulugvcse/national-flowers). It includes images in 9 classes: Tulip, Sunflower, Rose, Orchid, Lotus, Lilly, Lavender, Dandelion, Daisy.  The dataset is split into training and testing sets for model training and evaluation.
