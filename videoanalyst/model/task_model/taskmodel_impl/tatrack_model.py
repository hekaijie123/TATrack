# -*- coding: utf-8 -*

from loguru import logger

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.task_model.taskmodel_base import (TRACK_TASKMODELS,
                                                          VOS_TASKMODELS)

torch.set_printoptions(precision=8)


@TRACK_TASKMODELS.register
@VOS_TASKMODELS.register
class TATrack(ModuleBase):

    default_hyper_params = dict(pretrain_model_path="",
                                head_width=256,
                                conv_weight_std=0.01,
                                corr_fea_output=False,
                                amp=False)

    support_phases = ["train", "memorize", "track"]

    def __init__(self, backbone, neck, head, loss=None):   
        super(TATrack, self).__init__()
        self.basemodel = backbone
        # self.basemodel_q = backbone      ##
        self.neck = neck
        self.head = head
        self.loss = loss
        self._phase = "train"
        self.reset_parameters()

    def reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if self.neck.z_input_projection is not None:
            self.neck.z_input_projection.apply(_init_weights)
        if self.neck.x_input_projection is not None:
            self.neck.x_input_projection.apply(_init_weights)

        self.neck.encoder.apply(_init_weights)
        self.neck.decoder.apply(_init_weights)
        self.basemodel.encoder.apply(_init_weights)
        self.neck.out_norm.apply(_init_weights)
        self.neck.box_encoding.apply(_init_weights)


    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, p):
        assert p in self.support_phases
        self._phase = p

    def _track(self, z_feat, pre_z_feat, x_feat):
        z_pos = None
        x_pos = None

        if self.neck.z_pos_enc is not None:
            z_pos = self.neck.z_pos_enc().unsqueeze(0)
        if self.neck.x_pos_enc is not None:
            x_pos = self.neck.x_pos_enc().unsqueeze(0)

        z_feat, pre_z_feat, x_feat = self.neck.encoder(z_feat, pre_z_feat, x_feat, z_pos, x_pos)

        decoder_feat = self.neck.decoder(z_feat, pre_z_feat, x_feat, z_pos, x_pos)
        decoder_feat = self.neck.out_norm(decoder_feat)

        return self.head(decoder_feat)

    def memorize(self, im_crop, ltbr_label, gauss_label):
        fp = self.basemodel(im_crop,False)      #, fg_bg_label_map

        return fp

    def train_forward(self, training_data):
        memory_img = training_data["im_m"]
        query_img = training_data["im_q"]
        pre_img = training_data["im_p"]
        fm,fp,fq = self.basemodel(memory_img,pre_img,query_img)

        fp += self.neck.box_encoding(training_data["ltbr_label"],training_data["gauss_label"])

        return self._track(fm,fp,fq)

    def forward(self, *args, phase=None):
        if phase is None:
            phase = self._phase
        # used during training
        if phase == 'train':
            # resolve training data
            if self._hyper_params["amp"]:
                with torch.cuda.amp.autocast():
                    return self.train_forward(args[0])
            else:
                return self.train_forward(args[0])

        elif phase == 'memorize':
            target_img, ltbr_label, gauss_label  = args
            fp = self.memorize(target_img, ltbr_label, gauss_label)
            out_list = fp

        elif phase == 'track':

            fq, fm, fp, gauss_label, ltbr_label = args

            fp = fp.to(device = fm.device,dtype=torch.float)
            fm,fp,fq = self.basemodel(fm,fp,fq)
            ltbr_label = ltbr_label.to(device = fm.device,dtype=torch.float)
            gauss_label = gauss_label.to(device = fm.device,dtype=torch.float)           
            fp += self.neck.box_encoding(ltbr_label,gauss_label)

            out_list = self._track(fm,fp,fq)

        else:
            raise ValueError("Phase non-implemented.")

        return out_list

    def update_params(self):
        super().update_params()



    def set_device(self, dev):
        if not isinstance(dev, torch.device):
            dev = torch.device(dev)
        self.to(dev)
        if self.loss is not None:
            for loss_name in self.loss:
                self.loss[loss_name].to(dev)
