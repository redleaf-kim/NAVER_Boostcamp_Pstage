import math
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.cuda.amp.autocast_mode import autocast


def report(y_pred, y_true, cls_names=[str(_) for _ in range(18)]):
    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    
    # cls_report = classification_report(
    #     y_true=y_true,
    #     y_pred=y_pred,
    #     output_dict=True,
    #     target_names=cls_names,
    #     labels=np.arange(len(cls_names))
    # )
    
    matrix = confusion_matrix(y_pred, y_true)
    return matrix
    
# https://github.com/NingAnMe/Label-Smoothing-for-CrossEntropyLoss-PyTorch/blob/main/label_smothing_cross_entropy_loss.py
class LabelSmoothingCrossEntropy(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.2):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction
    
    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                                  device=targets.device) \
                .fill_(smoothing / (n_classes - 1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothingCrossEntropy._smooth_one_hot(targets, inputs.size(-1),
                                                              self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
            
        return loss
    
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=.25, eps=1e-7, weight=None, type='bce'):
        super(FocalLoss, self).__init__()
        
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        if type == 'bce':
            self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')
        elif type == 'smoothing':
            self.ce = LabelSmoothingCrossEntropy(weight=weight, reduction='none')
    
    def forward(self, inp, tar):
        with autocast():
            log_p = self.ce(inp, tar)
            p = torch.exp(-log_p)
            
            loss = self.alpha * (1-p) ** self.gamma * log_p
            return loss.mean()
    
    

class ArcFaceLoss(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, cfg, s=45.0, m=0.50, weight=None, reduction='mean'):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m

        self.reduction = reduction
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        if cfg.arcface_crit == 'bce':
            self.crit = nn.CrossEntropyLoss(weight=weight, reduction="none")
        elif cfg.arcface_crit == 'smoothing':
            self.crit = LabelSmoothingCrossEntropy(weight=weight, reduction='none')
        elif cfg.arcface_crit == 'focal':
            self.crit = FocalLoss(cfg.focal_gamma, type=cfg.focal_type)

    def forward(self, cosine, label):
        with autocast():
            # --------------------------- cos(theta) & phi(theta) ---------------------------
            sine = torch.sqrt((torch.sub(1.0, cosine*cosine)).clamp(0, 1))
            phi = torch.mul(cosine, self.cos_m) - torch.mul(sine, self.sin_m)
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
            
            loss = self.crit(output, label)
            if self.reduction == "mean": loss = loss.mean()
            elif self.reduction == "sum": loss = loss.sum()
            
            return loss