# -*- coding: utf-8 -*
from turtle import forward
import numpy as np
from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
from .iou.iou2d_calculator import bbox_overlaps

@TRACK_LOSSES.register
class GIOULoss(ModuleBase):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    default_hyper_params = dict(
        name="giouloss",
        weight = 1.0
    )

    def __init__(self, ):
        super().__init__()

    def forward(self, pred,target,eps=1e-6):
        num_pos = target['index_num'].sum()
        num_pos = max(num_pos.item(), 1e-4)
        pred, target,scores = self.trans_num(pred,target)
        gious,ious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
        loss = 1 - gious
        loss =(loss * scores).sum()
        weight = scores.sum()
        weight.clamp_(min=1.e-5)   
        iou = ious.detach().sum()/num_pos
        extra = dict(iou=iou)
        loss = self._hyper_params['weight']*loss/weight
        return loss,extra

    def trans_num(self, predicted, target) :    
        predicted_bbox =  predicted['bbox']
        cls_score = predicted['class_score'].detach().squeeze(1)
        index = np.nonzero(target["target_class_vector"])
        index = index.transpose(0,1)
        cls_score = cls_score[index[0],index[1],index[2]]
        predicted_bbox = predicted_bbox[index[0],index[1],index[2]]
        target_bbox = target['bbox_q'].repeat_interleave(target['index_num'].long(),axis=0)  
        return predicted_bbox,target_bbox,cls_score