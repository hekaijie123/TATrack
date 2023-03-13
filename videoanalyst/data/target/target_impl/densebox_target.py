# -*- coding: utf-8 -*-

from random import gauss
import torch
import numpy as np
from typing import Dict

from ..target_base import TRACK_TARGETS, TargetBase
from .utils.make_densebox_target import make_bbox_indices,gaussian_label_function,generate_ltbr_regression_targets

@TRACK_TARGETS.register
class DenseboxTarget(TargetBase):
    r"""
    Tracking data filter

    Hyper-parameters
    ----------------
    """
    default_hyper_params = dict(
        m_size=112,
        q_size=224,
        score_size=14,
        num_memory_frames=0,
    )

    def __init__(self) -> None:
        super().__init__()

    def update_params(self):
        hps = self._hyper_params
        self._hyper_params = hps

    def __call__(self, sampled_data: Dict) -> Dict:
        data_m = sampled_data["data1"]
        im_ms = []
        bbox_ms = []
        nmf = self._hyper_params['num_memory_frames']
        for i in range(nmf):
            im_ms.append(data_m['image_{}'.format(i)])
            bbox_ms.append(data_m['anno_{}'.format(i)])

        data_q = sampled_data["data2"]
        im_q, bbox_q = data_q["image"], data_q["anno"]

        # is_negative_pair = sampled_data["is_negative_pair"]

        # input tensor
        # im_m = np.stack(im_ms, axis=0)
        # im_m = np.squeeze(im_m,axis=0)
        im_m = im_ms[0]
        im_p = im_ms[1]
        bbox_ms = bbox_ms[1]         ###############要改
        # im_m = im_m.transpose(0, 3, 1, 2)  # T, C, H, W
        # im_q = im_q.transpose(2, 0, 1)
        gauss_label = gaussian_label_function(bbox_ms.view(1,-1),0.1,1,self._hyper_params['score_size'], self._hyper_params['q_size'], end_pad_if_even=True).view(-1,1)
        ltbr_label = generate_ltbr_regression_targets(bbox_ms.view(1,-1),16,self._hyper_params['q_size'])
        # training target
        target_bbox_feat_ranges,target_class_vector,index_num  = make_bbox_indices(
            bbox_q, self._hyper_params)
        bbox_q /= self._hyper_params['q_size']


        training_data = dict(
            im_m=im_m,
            im_q=im_q,
            im_p=im_p,
            bbox_q=bbox_q.to(torch.float),  
            index_num=index_num,
            target_bbox_feat_ranges = target_bbox_feat_ranges,
            target_class_vector = target_class_vector,
            gauss_label = gauss_label.to(torch.float),
            ltbr_label = ltbr_label.to(torch.float),
        )

        return training_data        #,lable
