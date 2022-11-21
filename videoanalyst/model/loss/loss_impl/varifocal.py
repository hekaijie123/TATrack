# -*- coding: utf-8 -*

import torch
import torch.nn.functional as F
from ...module_base import ModuleBase
from ..loss_base import TRACK_LOSSES
import numpy as np
from .iou.iou2d_calculator import bbox_overlaps

@TRACK_LOSSES.register
class VarifocalLoss(ModuleBase):

    default_hyper_params = dict(
        name="varifocaloss",
        alpha=0.75,
        gamma=2.0,
        weight=1.0,
        iou_weighted=True,
        use_sigmoid=False,
    )
  
    def __init__(self):
        """`Varifocal Loss <https://arxiv.org/abs/2008.13367>`_
        Args:
            use_sigmoid (bool, optional): Whether the prediction is
                used for sigmoid or softmax. Defaults to True.
            alpha (float, optional): A balance factor for the negative part of
                Varifocal Loss, which is different from the alpha of Focal
                Loss. Defaults to 0.75.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            iou_weighted (bool, optional): Whether to weight the loss of the
                positive examples with the iou target. Defaults to True.
        """
        super().__init__()

    def forward(self,
                pred,
                target):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
        Returns: 
            torch.Tensor: The calculated loss
        """
        num_pos = target['index_num'].sum()
        num_pos = max(num_pos.item(), 1e-4)
    
        pred,target = self.trans_num(pred,target)
        assert pred.size() == target.size()
        if self._hyper_params['use_sigmoid']:
            pred_sigmoid = pred.sigmoid()
        else:
            pred_sigmoid = pred
        target = target.type_as(pred)
        if self._hyper_params['iou_weighted']:
            focal_weight = target * (target > 0.0).float() + \
                self._hyper_params['alpha'] * (pred_sigmoid - target).abs().pow(self._hyper_params['gamma']) * \
                (target <= 0.0).float()
        else:
            focal_weight = (target > 0.0).float() + \
                self._hyper_params['alpha'] * (pred_sigmoid - target).abs().pow(self._hyper_params['gamma']) * \
                (target <= 0.0).float()
        loss = F.binary_cross_entropy(
            pred_sigmoid, target, reduction='none') * focal_weight
            
        loss = self._hyper_params['weight']* loss.sum()/num_pos   
        extra = dict() 
        return loss,extra




    def trans_num(self, predicted, target) :    
        cls_score, predicted_bbox = predicted['class_score'], predicted['bbox'].detach()
        N, num_classes, H, W = cls_score.shape
        quality_score = torch.zeros((N, H, W), dtype=torch.float, device=cls_score.device)

        index = np.nonzero(target["target_class_vector"])
        index = index.transpose(0,1)
        a = predicted_bbox[index[0],index[1],index[2]]
        b = target['bbox_q'].repeat_interleave(target['index_num'].long(),axis=0)  
        quality_score[index[0],index[1],index[2]] = bbox_overlaps(a,b,is_aligned=True).clamp(min=1e-6).to(torch.float)
        cls_score = cls_score.flatten(0)
        quality_score = quality_score.flatten(0)
        return cls_score,quality_score












