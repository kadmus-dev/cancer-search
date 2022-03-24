import pytorch_lightning as pl
import torch
import torchmetrics

from utils import object_from_dict
from loss import DiceLoss


class SegmentationModel(pl.LightningModule):
    def __init__(self, cfg):
        super(SegmentationModel, self).__init__()
        self._cfg = cfg
        self.model = object_from_dict(self._cfg.model)
        self.loss = DiceLoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):

        optimizer = object_from_dict(
            self._cfg.optimizer,
            params=[x for x in self.model.parameters() if x.requires_grad],
        )

        scheduler = object_from_dict(self._cfg.scheduler["lr_scheduler"], optimizer=optimizer)
        lr_dict = self._cfg.scheduler["lr_dict"]
        lr_dict["scheduler"] = scheduler

        return [optimizer], [lr_dict]

    def training_step(self, batch, batch_idx):
        features, gt = batch

        preds = self.forward(features)

        # calculating loss
        total_loss = self.loss(preds, gt)

        logs = {"train_loss": total_loss.detach()}

        # calculating metrics (for example)
        precision = torchmetrics.Precision()
        recall = torchmetrics.Recall()
        f1 = torchmetrics.F1()

        logs["precision"] = precision(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["recall"] = recall(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["f1"] = f1(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()

        return {"loss": total_loss, "logs": logs}

    def validation_step(self, batch, batch_idx):
        features, gt = batch

        preds = self.forward(features)

        logs = {"val_loss": self.loss(preds, gt).detach()}

        # calculating metrics (for example)
        precision = torchmetrics.Precision()
        recall = torchmetrics.Recall()
        f1 = torchmetrics.F1()

        logs["val_precision"] = precision(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["val_recall"] = recall(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()
        logs["val_f1"] = f1(preds.cpu(), gt.type(torch.IntTensor).cpu()).detach()

        return {"logs": logs}

    def training_epoch_end(self, outputs):
        logs = {
            "epoch": self.trainer.current_epoch,
            "train_loss": torch.stack([x['logs']['train_loss'] for x in outputs]).mean(),
            "precision": torch.stack([x['logs']['precision'] for x in outputs]).mean(),
            "recall": torch.stack([x['logs']['recall'] for x in outputs]).mean(),
            "f1": torch.stack([x['logs']['f1'] for x in outputs]).mean()
        }

        for name, value in logs.items():
            self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)

    def validation_epoch_end(self, outputs):
        logs = {
            "epoch": self.trainer.current_epoch,
            "val_loss": torch.stack([x['logs']['val_loss'] for x in outputs]).mean(),
            "val_precision": torch.stack([x['logs']['val_precision'] for x in outputs]).mean(),
            "val_recall": torch.stack([x['logs']['val_recall'] for x in outputs]).mean(),
            "val_f1": torch.stack([x['logs']['val_f1'] for x in outputs]).mean()
        }

        for name, value in logs.items():
            self.log(name, value, prog_bar=True, on_epoch=True, on_step=False)
