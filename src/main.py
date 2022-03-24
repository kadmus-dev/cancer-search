import os
from albumentations.core.serialization import from_dict
from torch.utils.data import DataLoader

from tissue_dataset import TissueDataset
from model import SegmentationModel
from utils import get_config, object_from_dict

CONFIG_PATH = os.path.join("src", "config.yaml")


def train():
    cfg = get_config(CONFIG_PATH)

    model = SegmentationModel(cfg)
    train_augs = from_dict(cfg.train_augs)
    ds = TissueDataset(cfg.img_dir, cfg.mask_dir, transforms=train_augs)
    dl = DataLoader(ds, **cfg.train_dl_args)

    trainer = object_from_dict(cfg.trainer)
    trainer.fit(model, train_dataloaders=dl, val_dataloaders=dl)


if __name__ == '__main__':
    train()
