import tensorflow as tf
from tensorflow.keras import layers

def get_image_preprocessing_pipeline(input_size, resize):
    image_preprocessing_pipeline = tf.keras.Sequential([
        layers.Resizing(height=resize, width=resize),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomCrop(int(resize*0.9), int(resize*0.9), 3),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(factor=0.1)
    ])
    image_preprocessing_pipeline.build(input_shape=(input_size, input_size, 3))
    return image_preprocessing_pipeline

import torch
from torchvision import transforms

def get_image_preprocessing_pipeline(input_size, resize):
    image_preprocessing_pipeline = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),  # 0.1 * 360 degrees
        transforms.RandomResizedCrop((int(resize*0.9), int(resize*0.9))),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return image_preprocessing_pipeline


import tensorflow as tf
from tensorflow.keras import layers

def get_image_preprocessing_pipeline(input_size, resize):
    image_preprocessing_pipeline = tf.keras.Sequential([
        layers.Resizing(height=resize, width=resize),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.1),
        layers.RandomContrast(0.1),
        layers.RandomCrop(int(resize*0.9), int(resize*0.9)),
        layers.RandomZoom(height_factor=0.1, width_factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomBrightness(factor=0.1)
    ])
    image_preprocessing_pipeline.build(input_shape=(input_size, input_size, 3))
    return image_preprocessing_pipeline
