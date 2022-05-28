import albumentations as A
import albumentations.pytorch as Ap
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000


class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, csv_path, imsize, is_train=True):

        self.data_path = data_path
        self.is_train = is_train
        self.data = pd.read_csv(csv_path)

        if is_train:
            self.augmentations = A.Compose([
                A.Resize(imsize, imsize),
                A.Flip(p=0.8),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), mask_value=(0, 0, 0)),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.5),
                A.Sharpen(),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0),
                Ap.ToTensorV2(),
            ])
        else:
            self.augmentations = A.Compose([
                A.Resize(imsize, imsize),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.), max_pixel_value=255.0),
                Ap.ToTensorV2(),
            ])

    def __getitem__(self, index):

        img_path = self.data_path + self.data.iloc[index]["name"] + ".jpg"
        mask_path = self.data_path + self.data.iloc[index]["name"] + "_mask.jpg"
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = np.round(mask / 255).astype(np.float32)

        aug_res = self.augmentations(image=image, mask=mask)
        img = aug_res["image"]
        mask = aug_res["mask"]

        return img, mask

    def __len__(self):
        return len(self.data)
