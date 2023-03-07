# coding=utf-8
# email: wangzejunscut@126.com

import torch
import torch.nn as nn
import torch.nn.functional as F

def binary_focal_loss(input, target, alpha=None, gamma=2, reduction="mean", pos_weight=None):
    # Compute focal loss for binary classification
    p = input.sigmoid()
    factor = ((1 - p) * target + p * (1 - target)).pow(gamma)
    if alpha is not None:
        factor = (alpha * target + (1 - alpha) * (1 - target)) * factor
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    loss = F.binary_cross_entropy(p, target, reduction="none", pos_weight=pos_weight) * factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss

def binary_label_smooth_loss(input, target, label_smoothing=0.1, reduction="mean", pos_weight=None):
    # Compute label smooth loss for binary classification
    if label_smoothing:
        target = target * (1 - label_smoothing) + 0.5 * label_smoothing
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)
    return F.binary_cross_entropy_with_logits(input, target, reduction=reduction, pos_weight=pos_weight)

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction="mean", pos_weight=None):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
    
    def forward(self, input, target):
        p = input.sigmoid()
        factor = ((1 - p) * target + p * (1 - target)).pow(self.gamma)
        if self.alpha is not None:
            factor = (self.alpha * target + (1 - self.alpha) * (1 - target)) * factor
        loss = F.binary_cross_entropy(p, target, reduction="none", pos_weight=self.pos_weight) * factor
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss

class BinaryLabelSmoothLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, reduction="mean", pos_weight=None):
        super(BinaryLabelSmoothLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.pos_weight = torch.tensor(pos_weight) if pos_weight is not None else None
    
    def forward(self, input, target):
        if self.label_smoothing:
            target = target * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return F.binary_cross_entropy_with_logits(input, target, reduction=self.reduction,
                pos_weight=self.pos_weight)
