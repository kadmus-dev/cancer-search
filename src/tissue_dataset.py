import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TissueDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = np.round(mask / 255).astype(np.float32)

        if self.transforms is not None:
            aug = self.transforms(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        return torch.from_numpy(image), torch.from_numpy(mask)
