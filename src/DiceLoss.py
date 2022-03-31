import torch


def generalized_dice_coefficient(y_true, y_pred):
    smooth = 1.
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (
            torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - generalized_dice_coefficient(y_true, y_pred)
    return loss
