import random
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import CenterCrop
import os
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import cv2


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha=0.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


# dice loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        intersection = (y_pred * y_true).sum()
        dice = (2.0 * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1 - dice


# evalution metri


class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = (
            np.sum(self.confusionMatrix, axis=1)
            + np.sum(self.confusionMatrix, axis=0)
            - np.diag(self.confusionMatrix)
        )
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        imgPredict = torch.sigmoid(imgPredict)
        imgPredict = torch.where(imgPredict >= 0.5, 1, 0)
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass**2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(
            imgPredict.cpu(), imgLabel.cpu()
        )

    def recall(self):
        # return all class recall
        # recall = TP / (TP + FN)
        recall = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=1)
        return recall

    def precision(self):
        # return all class precision
        # precision = TP / (TP + FP)
        precision = np.diag(self.confusionMatrix) / np.sum(self.confusionMatrix, axis=0)
        return precision

    def F1score(self):
        # return all class F1 score
        # F1 score = 2 * (precision * recall) / (precision + recall)
        precision = self.precision()
        recall = self.recall()
        F1score = 2 * (precision * recall) / (precision + recall)
        return F1score
