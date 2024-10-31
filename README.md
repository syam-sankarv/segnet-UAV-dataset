# README

## Project Overview

This project demonstrates a deep learning pipeline for semantic segmentation, focusing on the segmentation of images from a **UAV dataset of segetal flora**. The pipeline trains a custom segmentation model (based on **SegNet architecture**) to identify segetal flora in aerial imagery. While this code is tailored for a specific UAV dataset, it can be adapted for other datasets or proprietary image data with similar dimensions and characteristics. There are two scripts with and without data augmentation. Only specific techniques have been used for augmentation and can be changed according to needs.

**Note**: The UAV dataset used in this project is restricted for public use, and users must ensure compliance with applicable data access and usage policies if they utilize a different dataset.

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Data Preprocessing](#data-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Prediction and Visualization](#prediction-and-visualization)
6. [Performance Metrics](#performance-metrics)
7. [How to Run](#how-to-run)

---

### Environment Setup

#### Required Libraries

The code requires the following libraries:
- `numpy`, `pandas` - for data manipulation and analysis.
- `tensorflow`, `keras` - for building and training the neural network.
- `cv2` - for image processing.
- `matplotlib`, `seaborn` - for visualization.
- `sklearn` - for calculating performance metrics.

```python
import os
import cv2
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, Reshape, UpSampling2D, MaxPooling2D, concatenate
from tensorflow.keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
```

---

### Data Preprocessing

#### Loading the Dataset
Images and masks are loaded, resized to 512x512 pixels, and normalized to a range of `[0, 1]` for compatibility with the neural network. Masks are converted to binary format, assuming a single class for segmentation. 

```python
def load_dataset(image_dir, mask_dir, img_size):
    # Code to load, resize, and normalize images and masks.
```

#### Data Augmentation
Data augmentation is performed to enhance the model's generalization ability. Transformations include random rotation, translation, zoom, and horizontal flipping.

```python
data_gen_args = dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, ...)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
```

---

### Model Architecture

#### SegNet-Based Model
A custom SegNet-like model is implemented, which consists of an encoder-decoder structure. Convolutional and pooling layers are used for feature extraction, while up-sampling layers are employed to reconstruct the segmentation mask.

```python
def build_segnet(input_shape=(512, 512, 3)):
    # Code for encoder-decoder structure with Conv2D and UpSampling2D layers.
    model = Model(inputs=inputs, outputs=outputs)
    return model
```

---

### Training and Evaluation

#### Training the Model
The model is compiled with `Adam` optimizer and `binary cross-entropy` loss, suitable for binary segmentation. Key callbacks include:
- **ModelCheckpoint**: Saves the best model based on validation loss.
- **EarlyStopping**: Stops training if validation loss does not improve.

Visualization and timing callbacks are added to monitor training progress and measure training time per epoch.

```python
model.fit(all_images, all_masks, validation_data=(val_images, val_masks), batch_size=8, epochs=100, ...)
```

#### Performance Metrics
Metrics such as **Precision**, **F1 Score**, and **Mean IoU** are calculated on the validation dataset to assess model performance. These are saved to an Excel file for further analysis.

```python
precision = precision_score(val_masks.flatten(), val_preds_bin.flatten())
f1 = f1_score(val_masks.flatten(), val_preds_bin.flatten())
```

---

### Prediction and Visualization

#### Visualization Callback
Predictions on the validation dataset are visualized at the end of each epoch, comparing true and predicted masks for qualitative evaluation.

```python
class VisualizationCallback(tf.keras.callbacks.Callback):
    # Callback to visualize predictions at the end of each epoch.
```

#### Inference on New Images
The model is tested on new images, with predictions and associated inference times calculated and displayed. This step allows for model testing in a real-world application setting.

```python
start_time = time.time()
predictions = model.predict(sample_images)
inference_time = (end_time - start_time) / len(sample_images)
print(f"Average Inference Time per Image: {inference_time:.4f} seconds")
```

---

### Performance Metrics

#### Additional Metrics
Performance metrics are calculated and saved, and confusion matrix heatmaps and training accuracy/loss plots are generated for visual analysis.

```python
cm = confusion_matrix(val_masks.flatten(), val_preds_bin.flatten())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
```

---

### How to Run

1. Ensure all required libraries are installed.
2. Place dataset images and masks in the respective directories.
3. Run the script to train the model.
4. Review training history and metrics saved in the `training_history.xlsx` file.
5. Visualize predictions, training loss, and accuracy in the output plots.

#### Note
- Modify `train_image_dir`, `train_mask_dir`, `val_image_dir`, and `val_mask_dir` variables to point to your data.
- Results and predictions will be saved in the specified save directories, with generated images for each training epoch.

---
### References

- SegNet Paper: [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561)
- Original SegNet Repository: [Alex Kendall's SegNet on GitHub](https://github.com/alexgkendall/SegNet-Tutorial)

---

### License and Attribution
This repository contains a modified implementation of the SegNet model architecture. The original SegNet concept and model were developed by Alex Kendall et al., and all rights to the original work belong to the authors. This code is derived from open-source implementations, adapted to TensorFlow/Keras for educational purposes.

Please refer to the specific original `LICENSE` file for more information.

The original work can be found at [SegNet GitHub Repository](https://github.com/alexgkendall/SegNet-Tutorial), and modifications have been made to adapt the model to TensorFlow/Keras. All rights to the original work are reserved to the respective authors.

This guide should help set up and train the SegNet model for segmentation tasks. Further adjustments to the code (e.g., change of layers, adding data augmentation, tuning hyperparameters) can be made to improve model performance for specific datasets or applications.
