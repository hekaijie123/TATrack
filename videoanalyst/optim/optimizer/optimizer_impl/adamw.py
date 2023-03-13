# -*- coding: utf-8 -*-

import numpy as np
from yacs.config import CfgNode

import torch
from torch import optim

from videoanalyst.data.dataset.dataset_base import DatasetBase
from videoanalyst.evaluation.got_benchmark.datasets import got10k

from ..optimizer_base import OPTIMIZERS, OptimizerBase


@OPTIMIZERS.register
class AdamW(OptimizerBase):
    r"""
    Tracking data sampler

    Hyper-parameters
    ----------------
    """
    extra_hyper_params = dict(
        # lr=1e-4,
        weight_decay=1e-4,
    )

    def __init__(self, cfg: CfgNode, model: torch.nn.Module) -> None:
        super(AdamW, self).__init__(cfg, model)

    def update_params(self, ):
        super(AdamW, self).update_params()
        params = self._state["params"]
        kwargs = self._hyper_params
        valid_keys = self.extra_hyper_params.keys()
        kwargs = {k: kwargs[k] for k in valid_keys}
        self._optimizer = optim.AdamW(params, **kwargs)


AdamW.default_hyper_params.update(AdamW.extra_hyper_params)
