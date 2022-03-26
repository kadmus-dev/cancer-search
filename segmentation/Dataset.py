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
                # A.Flip(p=0.75),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
                Ap.ToTensorV2(),
            ])
        else:
            self.augmentations = A.Compose([
                A.Resize(imsize, imsize),
                A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
                Ap.ToTensorV2(),
            ])

    def __getitem__(self, index):

        img_path = self.data_path + self.data.iloc[index]["name"] + ".jpg"
        mask_path = self.data_path + self.data.iloc[index]["name"] + "_mask.jpg"
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        aug_res = self.augmentations(image=img, mask=mask)
        img = aug_res["image"]
        mask = aug_res["mask"].float().mean(axis=2)

        mask[mask < 128] = 0
        mask[mask >= 128] = 1

        return img, mask

    def __len__(self):
        return len(self.data)
