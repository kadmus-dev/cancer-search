import albumentations as A
import albumentations.pytorch as Ap
import cv2
import pandas as pd
import torch


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
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        aug_res = self.augmentations(image=image, mask=mask)
        image = aug_res["image"]
        mask = aug_res["mask"]

        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        mask = mask.int()

        return image, mask

    def __len__(self):
        return len(self.data)
