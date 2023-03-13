# encoding: utf-8
# import os
from typing import Dict, Tuple
import math
import numpy as np
# from videoanalyst.pipeline.utils.bbox import xyxy2cxywh
import torch

def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor, kernel_sz, feat_sz, image_sz, end_pad_if_even=True, density=False,
                            uni_bias=0):
    """Construct Gaussian label function."""

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = (target_bb[:, 0:2] +target_bb[:, 2:4]) * 0.5 
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label

def generate_ltbr_regression_targets( target_bb,stride,output_sz):
    shifts_x = torch.arange(
        0, output_sz, step=stride,
        dtype=torch.float32, device=target_bb.device
    )
    shifts_y = torch.arange(
        0, output_sz, step=stride,
        dtype=torch.float32, device=target_bb.device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    xs, ys = locations[:, 0], locations[:, 1]

    l = xs[:, None] - target_bb[:, 0][None]
    t = ys[:, None] - target_bb[:, 1][None]
    r = target_bb[:, 2][None] - xs[:, None]
    b = target_bb[:, 3][None] - ys[:, None]
    reg_targets_per_im = torch.stack([l, t, r, b], dim=2).reshape(-1, 4)

    reg_targets_per_im = reg_targets_per_im / output_sz


    # sz = output_sz//stride
    # nb = target_bb.shape[0]
    # reg_targets_per_im = reg_targets_per_im.reshape(sz, sz, nb, 4).permute(2, 3, 0, 1)

    return reg_targets_per_im


def make_bbox_indices(gt_boxes: np.array, config: Dict) -> Tuple:

    q_size = config["q_size"]
    score_size = config["score_size"]
    scale = score_size/q_size

    target_bbox_feat_ranges = gt_boxes.clone()
    target_bbox_feat_ranges[::2] = gt_boxes[::2] * scale
    target_bbox_feat_ranges[1::2] = gt_boxes[1::2] * scale
    target_bbox_feat_ranges = target_bbox_feat_ranges.to(torch.long)

    target_bbox_feat_ranges[3] += 1
    target_bbox_feat_ranges[2] += 1
    target_class_vector = torch.zeros([score_size , score_size], dtype=torch.float)
    target_class_vector[target_bbox_feat_ranges[1]: target_bbox_feat_ranges[3] , target_bbox_feat_ranges[0]: target_bbox_feat_ranges[2]]=1

    index_num = np.array(target_class_vector.sum())

    return target_bbox_feat_ranges,target_class_vector,index_num