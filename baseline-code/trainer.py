import os
import segmentation_models_pytorch as smp
import torch.nn as nn
import math, torch
import torchmetrics
from torchgeo.models.fcn import FCN
from torchgeo.trainers import SemanticSegmentationTask
from torch.optim.lr_scheduler import _LRScheduler
# from .soft_focal import FocalLoss
import numpy as np
from typing import Any, Dict, cast
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchgeo.datasets.utils import unbind_samples
import torch.nn.functional as F


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts,
              self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr) * self.step_in_cycle /
                    self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [
                base_lr + (self.max_lr - base_lr) *
                (1 + math.cos(math.pi *
                              (self.step_in_cycle - self.warmup_steps) /
                              (self.cur_cycle_steps - self.warmup_steps))) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (int(
                    (self.cur_cycle_steps - self.warmup_steps) *
                    self.cycle_mult) + self.warmup_steps)
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (epoch / self.first_cycle_steps *
                             (self.cycle_mult - 1) + 1),
                            self.cycle_mult,
                        ))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps *
                                                     (self.cycle_mult**n - 1) /
                                                     (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**(
                        n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class CELoss(nn.Module):
    def __init__(self, mode, ignore_index: int = 0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return self.ce(pred, target)


class DFC2022SemanticSegmentationTask(SemanticSegmentationTask):
    def config_task(self):
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "pan":
            self.model = smp.PAN(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "fcn":
            self.model = FCN(
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
                num_filters=self.hparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = CELoss(
                mode="multiclass",  # type: ignore[attr-defined]
                ignore_index=-1000 if self.ignore_zeros is None else 0,
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"])
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass",
                alpha=0.25,
                ignore_index=self.ignore_zeros,
                normalized=True,
                reduction="sum",
            )
        elif self.hparams["loss"] == "LovaszLoss":
            self.loss = smp.losses.LovaszClsLoss(
                mode="multiclass",
                ignore_index=0 if self.ignore_zeros else None,
                normalized=True,
            )
        else:
            raise ValueError(
                f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion_cps = nn.CrossEntropyLoss(reduction="mean",
                                                 ignore_index=0)
        self.train_metrics = torchmetrics.MetricCollection(
            {
                "OverallAccuracy":
                torchmetrics.Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "OverallPrecision":
                torchmetrics.Precision(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "OverallRecall":
                torchmetrics.Recall(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AverageAccuracy":
                torchmetrics.Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AveragePrecision":
                torchmetrics.Precision(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "AverageRecall":
                torchmetrics.Recall(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
                "IoU":
                torchmetrics.IoU(
                    num_classes=self.hparams["num_classes"],
                    ignore_index=self.ignore_zeros,
                ),
                "F1Score":
                torchmetrics.FBeta(
                    num_classes=self.hparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    mdmc_average="global",
                    ignore_index=self.ignore_zeros,
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    # 优化器修改
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.hparams["learning_rate"])

        scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                                  first_cycle_steps=40,
                                                  cycle_mult=1.0,
                                                  max_lr=1e-3,
                                                  min_lr=1e-6,
                                                  warmup_steps=10,
                                                  gamma=0.5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
