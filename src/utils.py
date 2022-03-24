import pydoc
import yaml
import torch
from collections import namedtuple
from PIL import Image
from torch.utils.data import DataLoader

from tissue_dataset import TissueDataset


Image.MAX_IMAGE_PIXELS = 933120000


def get_config(cfg_path: str) -> namedtuple:
    with open(cfg_path) as f:
        cfg_dict = yaml.load(f, yaml.SafeLoader)
    return namedtuple("Config", cfg_dict.keys())(**cfg_dict)


def object_from_dict(d, parent=None, **default_kwargs):
    kwargs = d.copy()
    object_type = kwargs.pop("type")
    for name, value in default_kwargs.items():
        kwargs.setdefault(name, value)

    if parent is not None:
        return getattr(parent, object_type)(**kwargs)  # skip PTC-W0034

    obj = pydoc.locate(object_type)
    if obj is None:
        raise ValueError(f"Bad object type: {object_type}")

    return obj(**kwargs)


def get_loader(
        val_dir,
        val_mask_dir,
        batch_size,
        val_transform,
        num_workers=4,
        pin_memory=True
):
    val_dataset = TissueDataset(
        image_dir=val_dir,
        mask_dir=val_mask_dir,
        transforms=val_transform
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return val_loader


def load_checkpoint(checkpoint, model):
    print('=> Loading checkpoints')
    model.load_state_dict(checkpoint['state_dict'])


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(f'Got acc {num_correct / num_pixels * 100:.2f}%')
    print(f'Dice score {dice_score / len(loader)}')

    model.train()

    return dice_score / len(loader)
