# -*- coding: utf-8 -*
import numpy as np

import torch
import torch.nn.functional as F

from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
from .utils import SafeLog

eps = np.finfo(np.float32).tiny


@TRACK_LOSSES.register
class SigmoidCrossEntropyCenterness(ModuleBase):
    default_hyper_params = dict(
        name="centerness",
        background=0,
        ignore_label=-1,
        weight=1.0,
    )

    def __init__(self, background=0, ignore_label=-1):
        super(SigmoidCrossEntropyCenterness, self).__init__()
        self.safelog = SafeLog()
        self.register_buffer("t_one", torch.tensor(1., requires_grad=False))

    def update_params(self, ):
        self.background = self._hyper_params["background"]
        self.ignore_label = self._hyper_params["ignore_label"]
        self.weight = self._hyper_params["weight"]

    def forward(self, pred_data, target_data):
        r"""
        Center-ness loss
        Computation technique originated from this implementation:
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        
        P.S. previous implementation can be found at the commit 232141cdc5ac94602c28765c9cf173789da7415e

        Arguments
        ---------
        pred: torch.Tensor
            center-ness logits (BEFORE Sigmoid)
            format: (B, HW)
        label: torch.Tensor
            training label
            format: (B, HW)

        Returns
        -------
        torch.Tensor
            scalar loss
            format: (,)
        """
        pred = pred_data["ctr_pred"]
        label = target_data["ctr_gt"]
        mask = (~(label == self.background)).type(torch.Tensor).to(pred.device)
        loss = F.binary_cross_entropy_with_logits(pred, label,
                                                  reduction="none") * mask
        # suppress loss residual (original vers.)
        loss_residual = F.binary_cross_entropy(label, label,
                                               reduction="none") * mask
        loss = loss - loss_residual.detach()

        loss = loss.sum() / torch.max(mask.sum(),
                                      self.t_one) * self._hyper_params["weight"]
        extra = dict()

        return loss, extra



