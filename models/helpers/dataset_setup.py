import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import torch.nn as nn
import albumentations as A
import numpy as np
import os
import time
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

TRAIN_FOLDER = "../../train"

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[...,None],[1,1,3])
    img = img.astype('float32')
    mx = np.max(img)
    if mx:
        img/=mx
    img = np.transpose(img,(2,0,1))
    img_ten = torch.tensor(img)
    return img_ten

def preprocess_mask(path):
    
    msk = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    msk = msk.astype('float32')
    msk/=255.0
    msk_ten = torch.tensor(msk)
    
    return msk_ten

class CustomDataset(Dataset):
    def __init__(self,image_files, mask_files, input_size=(256, 256), augmentation_transforms=None):
        self.image_files=image_files
        self.mask_files=mask_files
        self.input_size=input_size
        self.augmentation_transforms=augmentation_transforms
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path=self.image_files[idx]
        mask_path=self.mask_files[idx]
        
        image = preprocess_image(image_path)
        mask = preprocess_mask(mask_path)
        if self.augmentation_transforms:
            image,mask=self.augmentation_transforms(image, mask, self.input_size)
        return image, mask

def augment_image(image, mask, input_size):
    
    image_np = image.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()

    transform = A.Compose([
        A.Resize(height=input_size[0], width=input_size[1], interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=45, shift_limit=0.1, p=0.8, border_mode=0),
        A.RandomCrop(height=input_size[0], width=input_size[1], p=0.8),
        A.RandomBrightness(p=0.9),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=0.6),
                A.MotionBlur(blur_limit=3, p=0.4),
            ],
            p=0.9,
        ),
    
    ])
    augmented = transform(image = image_np,mask = mask_np)
    augmented_image, augmented_mask = augmented['image'], augmented['mask']
    
    augmented_image = torch.tensor(augmented_image, dtype=torch.float32).permute(2, 0, 1)
    augmented_mask  = torch.tensor(augmented_mask,dtype=torch.float32)
    
    return augmented_image, augmented_mask