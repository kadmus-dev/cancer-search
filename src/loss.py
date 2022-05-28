import torch
from torch.nn import functional as F


def dice_loss(y_true, y_pred, smooth=1):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    return 1 - (2. * (y_true * y_pred).sum() + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def iou_loss(y_true, y_pred, smooth=1):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    intersection = (y_pred * y_true).sum()
    return 1 - (intersection + smooth) / ((y_pred + y_true).sum() - intersection + smooth)


def dice_bce_loss(y_true, y_pred, smooth=1):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    return F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float(), reduction='mean') + \
           dice_loss(y_true, y_pred, smooth)


def focal_loss(y_true, y_pred, smooth=1, alpha=0.8, gamma=2):
    y_pred = y_pred.view(-1).float()
    y_true = y_true.view(-1).float()
    BCE = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
    return alpha * (1 - torch.exp(-BCE)) ** gamma * BCE


def tversky_loss(y_true, y_pred, smooth=1, alpha=0.5, beta=0.5):
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    TP = (y_pred * y_true).sum()
    FP = ((1 - y_true) * y_pred).sum()
    FN = (y_true * (1 - y_pred)).sum()
    return 1 - (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
