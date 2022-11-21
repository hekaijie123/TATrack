
from videoanalyst.model.backbone.backbone_base import (TRACK_BACKBONES,
                                                       VOS_BACKBONES)
from videoanalyst.model.module_base import ModuleBase
from models.backbone.swin_transformer import *
from models.methods.TATrack.modules.encoder.builder import build_encoder
from models.utils.drop_path import DropPathAllocator
@VOS_BACKBONES.register
@TRACK_BACKBONES.register
class base(ModuleBase):

    default_hyper_params = dict(
        name = "swin_tiny_patch4_window7_224",
        output_layers = [2],
    )

    def __init__(self):
        super(base, self).__init__()

    def update_params(self):
        super().update_params()
        self.backbone = build_swin_transformer_backbone(load_pretrained= True, **self._hyper_params)

        self.window_size = 7
        self.shift_size = self.window_size // 2
        self.config = dict(position_embedding=False,
            absolute = True,
            relative = True,
            transformer_type = "concatenation_feature_fusion",
            encoder_layer =8,
            with_branch_index=True, )
        drop_path_allocator = DropPathAllocator(0.1)
        with drop_path_allocator:
            self.encoder = build_encoder(self.config, drop_path_allocator,
                                        512, 8, 4.0, True, 0.0, 0.0,[7,7], [14,14])
        # self.norm2 = nn.LayerNorm(384)


    def encoder_attention (self, z, pre_z, x, z_pos, x_pos,i):
        concatenated = torch.cat((z, pre_z, x), dim=1)

        attn_pos_enc = None
        if self.encoder.z_untied_pos_enc is not None:
            z_q_pos, z_k_pos = self.encoder.z_untied_pos_enc()
            pre_z_q_pos,pre_z_k_pos = self.encoder.pre_z_untied_pos_enc()
            x_q_pos, x_k_pos = self.encoder.x_untied_pos_enc()
            attn_pos_enc = (torch.cat((z_q_pos, pre_z_q_pos, x_q_pos), dim=1) @ torch.cat((z_k_pos, pre_z_k_pos, x_k_pos), dim=1).transpose(-2, -1)).unsqueeze(0)

        if self.encoder.rpe_bias_table is not None:
            if attn_pos_enc is not None:
                attn_pos_enc = attn_pos_enc + self.encoder.rpe_bias_table(self.encoder.rpe_index)       ##rpe_bias_table 8dim * max_position_index  learnable
            else:
                attn_pos_enc = self.encoder.rpe_bias_table(self.encoder.rpe_index)


        # tenper_mask = torch.zeros_like(attn_pos_enc)
        # tenper_mask[::,:z.shape[1],z.shape[1]:z.shape[1]+x.shape[1]] = -100.0
        # attn_pos_enc += tenper_mask

        concatenated_pos_enc = None
        if z_pos is not None:
            assert x_pos is not None
            concatenated_pos_enc = torch.cat((z_pos, x_pos), dim=1)
        concatenated = self.encoder.layers[i](concatenated, concatenated_pos_enc, concatenated_pos_enc, attn_pos_enc)
        return concatenated[:, :z.shape[1], :], concatenated[:, z.shape[1]:z.shape[1]+pre_z.shape[1], :], concatenated[:, z.shape[1]+pre_z.shape[1]:, :]        

    def attention_mask(self,H,W,device):
        Hp = int(math.ceil(H / self.window_size)) * self.window_size
        Wp = int(math.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  
        return attn_mask      

    def extract_featrue(self,x):
        _, _, H, W = x.size()
        x, H, W = self.backbone.stages[0](x, H, W)
        x, H, W = self.backbone.stages[1](x, H, W)
        x, H, W = self.backbone.stages[2].pre_stage(x, H, W)
        return x,H,W

    def forward(self, z, pre_z, x, reshape=True):


        x, H, W = self.extract_featrue(x)
        pre_z, H, W = self.extract_featrue(pre_z)
        z, H1, W1 = self.extract_featrue(z)

        attn_mask_x = self.attention_mask(H,W,x.device)
        attn_mask_z = self.attention_mask(H1,W1,x.device)

        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # for i,(blk_x,blk_z) in enumerate(zip(self.backbone.stages[2].blocks,self.blocks)):
        #     blk_x.H, blk_x.W = H, W
        #     x = blk_x(x, attn_mask_x)
        #     blk_x.H, blk_x.W = H, W
        #     pre_z = blk_x(pre_z, attn_mask_x)
        #     blk_z.H, blk_z.W = H1, W1
        #     z = blk_z(z, attn_mask_z)
        for i,blk in enumerate(self.backbone.stages[2].blocks):
            blk.H, blk.W = H, W
            x = blk(x, attn_mask_x)
            blk.H, blk.W = H, W
            pre_z = blk(pre_z, attn_mask_x)
            blk.H, blk.W = H1, W1
            z = blk(z, attn_mask_z)

            if(((i+1)%2 == 0)and((i+1) != len(self.backbone.stages[2].blocks))):
                z,pre_z,x = self.encoder_attention(z,pre_z,x,None,None,int(i/2))

        pre_z = self.backbone.norm2(pre_z)
        x = self.backbone.norm2(x)
        z = self.backbone.norm2(z)
        # z = self.norm2(z)
        return z,pre_z,x