
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from functools import partial
from ..core import register
from .vit_tiny import VisionTransformer


import torch
import torch.nn as nn

def _gn_groups(num_channels, default_groups=8):
    
    return default_groups if (num_channels % default_groups == 0) else 1

class DWConvBlockRes(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, gn_groups=8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(out_ch, gn_groups), out_ch)
        self.act = nn.GELU()

        self.use_proj = (stride != 1) or (in_ch != out_ch)
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, 0, bias=False),
                nn.GroupNorm(_gn_groups(out_ch, gn_groups), out_ch),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.norm(y)
        y = self.act(y)
        return self.proj(x) + y


class SELayerLS(nn.Module):
    def __init__(self, ch, r=8, init_gamma=0.0):
        super().__init__()
        mid = max(1, ch // r)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, mid, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(mid, ch, 1, bias=True),
            nn.Sigmoid()
        )
        # LayerScale: per-channel
        self.gamma = nn.Parameter(torch.full((1, ch, 1, 1), float(init_gamma)))

    def forward(self, x):
        w = self.fc(x)                 # [B,C,1,1] in (0,1)
        return x * (1.0 + self.gamma * (w - 0.5))


class FusionAtt(nn.Module):
    def __init__(self, sem_ch, det_ch):
        super().__init__()
        ch = sem_ch + det_ch
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch//4, 1),
            nn.GELU(),
            nn.Conv2d(ch//4, ch, 1),
            nn.Sigmoid()
        )
    def forward(self, sem, det):
        x = torch.cat([sem, det], dim=1)
        w = self.fc(x)
        w_sem, w_det = torch.split(w, [sem.shape[1], det.shape[1]], dim=1)
        return sem * w_sem + det * w_det

class SpatialPriorModulev3(nn.Module):
    def __init__(self, inplanes=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, inplanes, 3, 2, 1, bias=False),
            nn.GroupNorm(_gn_groups(inplanes), inplanes),
            nn.GELU(),
            nn.Conv2d(inplanes, inplanes, 3, 2, 1, groups=inplanes, bias=False),
            nn.Conv2d(inplanes, inplanes, 1, 1, 0, bias=False),
            nn.GroupNorm(_gn_groups(inplanes), inplanes),
            nn.GELU(),
        )

        self.conv2 = DWConvBlockRes(inplanes, 2*inplanes, stride=2)
        self.se2   = SELayerLS(2*inplanes, init_gamma=0.0)

        self.conv3 = DWConvBlockRes(2*inplanes, 4*inplanes, stride=2)
        self.se3   = SELayerLS(4*inplanes, init_gamma=0.0)

        self.conv4 = DWConvBlockRes(4*inplanes, 4*inplanes, stride=2)
        self.se4   = SELayerLS(4*inplanes, init_gamma=0.0)

    def forward(self, x):
        c1 = self.stem(x)     # 1/4
        c2 = self.se2(self.conv2(c1))  # 1/8
        c3 = self.se3(self.conv3(c2))  # 1/16
        c4 = self.se4(self.conv4(c3))  # 1/32
        return c2, c3, c4



@register()
class DINOv3SPM(nn.Module):
    def __init__(
        self,
        name=None,
        weights_path=None,
        interaction_indexes=[],
        finetune=True,
        embed_dim=192,
        num_heads=3,
        patch_size=16,
        use_sta=True,
        conv_inplane=16,
        hidden_dim=None,
    ):
        super(DINOv3SPM, self).__init__()
        if 'dinov3' in name:
            self.dinov3 = torch.hub.load('./dinov3', name, source='local', weights=weights_path)
            while len(self.dinov3.blocks) != (interaction_indexes[-1] + 1):
                del self.dinov3.blocks[-1]
            del self.dinov3.head
        else:
            self.dinov3 =  VisionTransformer(embed_dim=embed_dim, num_heads=num_heads)
            if weights_path is not None:
                print(f'Loading ckpt from {weights_path}...')
                checkpoint = torch.load(weights_path)
                self.dinov3._model.load_state_dict(checkpoint)
            else:
                print('Training ViT-Tiny from scratch!')

        embed_dim = self.dinov3.embed_dim
        self.interaction_indexes = interaction_indexes
        self.patch_size = patch_size

        if not finetune:
            self.dinov3.eval()
            self.dinov3.requires_grad_(False)

        # init the feature pyramid
        self.use_sta = use_sta
        if use_sta:
            print(f"Using Lite Spatial Prior Module with inplanes={conv_inplane}")
            self.sta = SpatialPriorModulev3(inplanes=conv_inplane)
        else:
            conv_inplane = 0

        # linear projection
        hidden_dim = hidden_dim if hidden_dim is not None else embed_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(embed_dim + conv_inplane*2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(embed_dim + conv_inplane*4, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
        ])
        # norm
        self.norms = nn.ModuleList([
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim),
            nn.SyncBatchNorm(hidden_dim)
        ])

    def forward(self, x):
        # Code for matching with oss
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        H_toks, W_toks = x.shape[2] // self.patch_size, x.shape[3] // self.patch_size
        bs, C, h, w = x.shape

        if len(self.interaction_indexes) > 0 and not isinstance(self.dinov3, VisionTransformer):
            all_layers = self.dinov3.get_intermediate_layers(
                x, n=self.interaction_indexes, return_class_token=True
            )
        else:
            all_layers = self.dinov3(x)

        if len(all_layers) == 1:    # repeat the same layer for all the three scales
            all_layers = [all_layers[0], all_layers[0], all_layers[0]]
        
        sem_feats = []
        num_scales = len(all_layers) - 2
        for i, sem_feat in enumerate(all_layers):
            feat, _ = sem_feat
            sem_feat = feat.transpose(1, 2).view(bs, -1, H_c, W_c).contiguous()  # [B, D, H, W]
            resize_H, resize_W = int(H_c * 2**(num_scales-i)), int(W_c * 2**(num_scales-i))
            sem_feat = F.interpolate(sem_feat, size=[resize_H, resize_W], mode="bilinear", align_corners=False)
            sem_feats.append(sem_feat)

        # fusion
        fused_feats = []
        if self.use_sta:
            detail_feats = self.sta(x)
            for sem_feat, detail_feat in zip(sem_feats, detail_feats):
                fused_feats.append(torch.cat([sem_feat, detail_feat], dim=1))
        else:
            fused_feats = sem_feats

        c2 = self.norms[0](self.convs[0](fused_feats[0]))
        c3 = self.norms[1](self.convs[1](fused_feats[1]))
        c4 = self.norms[2](self.convs[2](fused_feats[2]))

        return c2, c3, c4
