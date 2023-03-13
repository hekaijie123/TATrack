import torch
import torch.nn as nn
import math
from videoanalyst.model.module_base import ModuleBase
from videoanalyst.model.neck.neck_base import TRACK_NECKS

from models.utils.drop_path import DropPathAllocator
import torch.nn as nn
from models.methods.TATrack.modules.encoder.builder import build_encoder
from models.methods.TATrack.modules.decoder.builder import build_decoder
from models.methods.TATrack.positional_encoding.builder import build_position_embedding
from timm.models.layers import  trunc_normal_

@TRACK_NECKS.register
class Transformer(ModuleBase):
    default_hyper_params = dict(
        position_embedding = False,
        position_type = "sine",
        with_branch_index = True,
        absolute = True,
        relative = True, 
        drop_path_rate =0.1,
        backbone_dim = 384,
        transformer_dim = 384,
        z_shape = [14,14],
        x_shape = [14,14],
        num_heads = 8,
        mlp_ratio = 4,
        qkv_bias = True,
        drop_rate = 0,
        attn_drop_rate = 0,
        transformer_type = "concatenation_feature_fusion",
        encoder_layer = 4,
        decoder_layer = 1,

    )

    def __init__(self):
        super(Transformer, self).__init__()

    # def forward(self, x):
    #     return self.adjustor(x)

    def update_params(self):
        super().update_params()
        self.position_embedding = self._hyper_params['position_embedding']
        self.position_type = self._hyper_params['position_type']
        self.with_branch_index = self._hyper_params['with_branch_index']
        self.absolute = self._hyper_params['absolute']
        self.relative = self._hyper_params['relative']
        self.transformer_type = self._hyper_params['transformer_type']
        self.encoder_layer = self._hyper_params['encoder_layer']
        self.decoder_layer = self._hyper_params['decoder_layer']
        
        self.z_shape = self._hyper_params['z_shape']
        self.x_shape = self._hyper_params['x_shape']
        self.num_heads = self._hyper_params['num_heads']
        self.mlp_ratio = self._hyper_params['mlp_ratio']
        self.qkv_bias = self._hyper_params['qkv_bias']
        self.drop_rate = self._hyper_params['drop_rate']
        self.backbone_dim = self._hyper_params['backbone_dim']
        self.attn_drop_rate = self._hyper_params['attn_drop_rate']
        self.transformer_dim = self._hyper_params['transformer_dim']
        self.drop_path_rate = self._hyper_params['drop_path_rate']

        drop_path_allocator = DropPathAllocator(self.drop_path_rate)
        self.z_input_projection = None
        self.x_input_projection = None
        if self.backbone_dim != self.transformer_dim:
            self.z_input_projection = nn.Linear(self.backbone_dim, self.transformer_dim)
            self.x_input_projection = nn.Linear(self.backbone_dim, self.transformer_dim)
        
        self.config = dict(position_embedding=self.position_embedding,
            absolute = self.absolute,
            relative=self.relative,
            transformer_type = self.transformer_type,
            encoder_layer =self.encoder_layer,
            decoder_layer = self.decoder_layer,
            with_branch_index=self.with_branch_index, )
        self.z_pos_enc, self.x_pos_enc = build_position_embedding(self.config , self.z_shape, self.x_shape, self.transformer_dim)    
        with drop_path_allocator:
            self.encoder = build_encoder(self.config, drop_path_allocator,
                                    self.transformer_dim, self.num_heads, self.mlp_ratio, self.qkv_bias, self.drop_rate, self.attn_drop_rate,
                                    self.z_shape, self.x_shape)

            self.decoder = build_decoder(self.config, drop_path_allocator,
                                    self.transformer_dim, self.num_heads, self.mlp_ratio, self.qkv_bias, self.drop_rate, self.attn_drop_rate,
                                    self.z_shape, self.x_shape)

        self.out_norm = nn.LayerNorm(self.transformer_dim)
        # self.memory_fuse = memory_fuse()
        self.box_encoding = embed_box(4,64,self.backbone_dim)



class memory_fuse(nn.Module):
    def __init__(self):
        super(memory_fuse, self).__init__()

    def forward(self, fm, fq):
        B, C, T, H, W = fm.size()
        fm0 = fm.clone()
        # fq0 = fq.clone()

        fm = fm.view(B, C, T * H * W)  # B, C, THW
        fm = torch.transpose(fm, 1, 2)  # B, THW, C
        fq = fq.view(B, C, H * W)  # B, C, HW

        w = torch.bmm(fm, fq) / math.sqrt(C)  # B, THW, HW
        w = torch.softmax(w, dim=1)

        fm1 = fm0.view(B, C, T * H * W)  # B, C, THW
        mem_info = torch.bmm(fm1, w)  # (B, C, THW) x (B, THW, HW) = (B, C, HW)
        mem_info = mem_info.view(B, C, H, W)

        # y = torch.cat([mem_info, fq0], dim=1)    合并
        y = mem_info.view(B,C,H*W).transpose(1,2)
        return y

class embed_box(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        self.Linear1 = nn.Linear(in_features,hidden_features)
        self.Linear2 = nn.Linear(hidden_features,out_features)
        self.Linear3 = nn.Linear(out_features,out_features)
        self.weight = nn.Parameter(torch.empty(1, 1, out_features))
        self.act = nn.GELU()
        trunc_normal_(self.weight, std=.02)

    def forward(self, ltrb_label, gauss_label):
        ltrb_label = self.Linear1(ltrb_label)
        ltrb_label = self.act(ltrb_label)
        ltrb_label = self.Linear2(ltrb_label)
        ltrb_label = self.act(ltrb_label)
        
        return self.Linear3(ltrb_label) + gauss_label*self.weight

