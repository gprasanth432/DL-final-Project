# DL-final-Project
This repository is for CS 676 Deep Learning final project

# Weapon Detection - Gun and Knife Classification

This project builds a simple deep learning pipeline to classify cropped weapon images (guns vs knives) from annotated datasets in YOLO format.

## Dataset Setup

- The dataset is obtained from kaggle and  stored in a ZIP file on Google Drive.
- The structure after extraction:

```
/content/data/knivesnpistols/
    └── combined_gunsnknifes/
        ├── train/
        │   ├── images/
        │   └── labels/
        └── val/
            ├── images/
            └── labels/
```

## Pipeline Steps

1. **Google Drive Mounting & Dataset Extraction**
   - Mounts the drive and extracts the dataset zip to the working directory.

2. **YOLO Label Parsing**
   - Parses YOLO `.txt` annotation files to extract bounding boxes and class labels.
   - Crops images using the bounding boxes for training.

3. **Custom PyTorch Dataset: `WeaponCropDataset`**
   - Applies transforms (resize to 32x32, tensor conversion).
   - Returns cropped image regions and labels for classification.

4. **Dataloader Preparation**
   - Loads the processed crops into `DataLoader` objects for training and validation.

5. **Model: `TinyCNN`**
   - A small CNN with:
     - 1 Conv2D layer
     - ReLU + MaxPool
     - Flatten + FC layers + Dropout
   - Designed to classify cropped weapon images into two classes.

##  Model Summary

```
TinyCNN(
  (features): Sequential(
    (0): Conv2d(3, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Flatten()
    (1): Linear(in_features=1024, out_features=16, bias=True)
    (2): ReLU()
    (3): Dropout(p=0.5, inplace=False)
    (4): Linear(in_features=16, out_features=2, bias=True)
  )
)
```

Update :
Model creation and training is done, model class created, model training function created, model trained
