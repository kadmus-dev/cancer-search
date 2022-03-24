import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from model import SegmentationModel
from utils import get_loader, load_checkpoint, check_accuracy, get_config
from main import CONFIG_PATH


def main():

    val_transforms = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    cfg = get_config(CONFIG_PATH)

    model = SegmentationModel(cfg)
    load_checkpoint(torch.load(f'weights/{model}'), model)
    model = model.to(config.DEVICE)

    val_loader = get_loader(
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        val_transforms,
        config.NUM_WORKERS,
        config.PIN_MEMORY
    )
    
    check_accuracy(val_loader, model, device=config.DEVICE)


if __name__ == '__main__':
    main()
