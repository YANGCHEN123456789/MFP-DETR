
import copy
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import get_activation
from ..core import register

__all__ = ['HybridEncoder']


class ConvNormLayer_fuse(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

        # for fuse
        self.ch_in, self.ch_out, self.kernel_size, self.stride, self.g, self.padding, self.bias = \
            ch_in, ch_out, kernel_size, stride, g, padding, bias

    def forward(self, x):
        if hasattr(self, 'conv_bn_fused'):
            y = self.conv_bn_fused(x)
        else:
            y = self.norm(self.conv(x))
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv_bn_fused'):
            self.conv_bn_fused = nn.Conv2d(
                self.ch_in, self.ch_out, self.kernel_size, self.stride,
                groups=self.g, padding=self.padding, bias=True
            )
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv_bn_fused.weight.data = kernel
        self.conv_bn_fused.bias.data = bias
        self.__delattr__('conv')
        self.__delattr__('norm')

    def get_equivalent_kernel_bias(self):
        kernel, bias = self._fuse_bn_tensor()
        return kernel, bias

    def _fuse_bn_tensor(self):
        kernel = self.conv.weight
        running_mean = self.norm.running_mean
        running_var = self.norm.running_var
        gamma = self.norm.weight
        beta = self.norm.bias
        eps = self.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, g=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.conv = nn.Conv2d(
            ch_in, ch_out, kernel_size, stride,
            groups=g, padding=padding, bias=bias
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s, act=None):
        super().__init__()
        self.cv1 = ConvNormLayer_fuse(c1, c2, 1, 1)
        self.cv2 = ConvNormLayer_fuse(c2, c2, k, s, c2)

    def forward(self, x):
        return self.cv2(self.cv1(x))


class VGGBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
        return self.act(y)


class CSPLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock):
        super(CSPLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer_fuse(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_2 = self.conv2(x)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        return self.conv3(x_1 + x_2)


class CSPLayer2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=False,
                 act="silu",
                 bottletype=VGGBlock,
                 ):
        super(CSPLayer2, self).__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv1 = ConvNormLayer_fuse(in_channels, hidden_channels * 2, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            bottletype(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer_fuse(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        y = list(self.conv1(x).chunk(2, 1))
        return self.conv3(y[0] + self.bottlenecks(y[1]))


class RepNCSPELAN4(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3,
                 bias=False,
                 act="silu",
                 csp_type='csp2',
                 ):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        CSPL = CSPLayer2 if csp_type == 'csp2' else CSPLayer
        self.cv2 = nn.Sequential(
            CSPL(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv3 = nn.Sequential(
            CSPL(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock),
            ConvNormLayer_fuse(c4, c4, 3, 1, bias=bias, act=act)
        )
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class RepNCSPELAN5(nn.Module):
    def __init__(self, c1, c2, c3, c4, n=3, bias=False, act="silu"):
        super().__init__()
        self.c = c3 // 2
        self.cv1 = ConvNormLayer_fuse(c1, c3, 1, 1, bias=bias, act=act)
        self.cv2 = nn.Sequential(CSPLayer2(c3 // 2, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv3 = nn.Sequential(CSPLayer2(c4, c4, n, 1, bias=bias, act=act, bottletype=VGGBlock))
        self.cv4 = ConvNormLayer_fuse(c3 + (2 * c4), c2, 1, 1, bias=bias, act=act)

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        out = self.cv4(torch.cat(y, 1))
        return out


# ============================
# Wavelet-Fourier cooperative enhancement (ONLY S3/S4)
# ============================
class WaveletFourierCoop(nn.Module):
    """
    Lightweight Wavelet (Haar-like) + Fourier cooperative enhancement.

    Key fixes for training stability:
      1) Fourier FFT is computed in FP32 to avoid cuFFT half-precision limitation
         on non power-of-two shapes (e.g., 80x80).
      2) Global pooling uses mean(...) to avoid calflops patch keyword issues.
    """

    def __init__(self, channels: int, act: str = "silu", fourier_init: float = 0.0):
        super().__init__()
        self.channels = channels
        self.act = get_activation(act)

        k = torch.tensor([
            [[1.,  1.],
             [1.,  1.]],
            [[1.,  1.],
             [-1., -1.]],
            [[1., -1.],
             [1., -1.]],
            [[1., -1.],
             [-1., 1.]],
        ]) / 2.0
        self.register_buffer("haar_kernels", k[None, ...])  # [1,4,2,2]

        self.wavelet_mix = nn.Conv2d(channels * 4, channels, kernel_size=1, bias=False)
        self.wavelet_bn = nn.BatchNorm2d(channels)

        self.fourier_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        nn.init.constant_(self.fourier_gate[0].weight, fourier_init)
        nn.init.constant_(self.fourier_gate[0].bias, 0.0)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def _wavelet_branch(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        k = self.haar_kernels.to(dtype=x.dtype, device=x.device)      # [1,4,2,2]
        k = k.repeat(C, 1, 1, 1).reshape(C * 4, 1, 2, 2)             # [C*4,1,2,2]
        y = F.conv2d(x, k, bias=None, stride=1, padding=1, groups=C)  # (B,C*4,H+1,W+1)
        y = y[..., :H, :W]
        y = self.wavelet_mix(y)
        y = self.wavelet_bn(y)
        return self.act(y)

    def _fourier_branch(self, x: torch.Tensor) -> torch.Tensor:
        """
        FP32 FFT to avoid:
          RuntimeError: cuFFT only supports dimensions whose sizes are powers of two
          when computing in half precision, but got [80,80]
        """
        orig_dtype = x.dtype
        x32 = x.float()

        X = torch.fft.rfft2(x32, dim=(-2, -1), norm="ortho")  # complex64
        amp = torch.abs(X)                                    # float32
        phase = X / (amp + 1e-6)

        amp_pool = amp.mean(dim=(-2, -1), keepdim=True)       # (B,C,1,1)
        gate = self.fourier_gate(amp_pool)                    # (B,C,1,1), float32

        amp2 = amp * (1.0 + gate)
        X2 = phase * amp2
        y32 = torch.fft.irfft2(X2, s=x32.shape[-2:], dim=(-2, -1), norm="ortho")  # float32

        return y32.to(dtype=orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wv = self._wavelet_branch(x)
        fr = self.act(self._fourier_branch(x))

        g = self.fuse(torch.cat([x, wv, fr], dim=1))
        y = x + g * (wv + fr)
        y = y + self.refine(y)
        return self.act(y)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)
        if self.norm is not None:
            output = self.norm(output)
        return output


@register()
class HybridEncoder(nn.Module):
    __share__ = ['eval_spatial_size', ]

    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None,
                 version='dfine',
                 csp_type='csp',
                 fuse_op='cat',
                 # NEW
                 use_wfcoop=True,
                 wfcoop_idx=(0, 1),  # only S3,S4
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.fuse_op = fuse_op

        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            if in_channel != hidden_dim:
                proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            else:
                proj = nn.Identity()
            self.input_proj.append(proj)

        # Wavelet-Fourier coop (ONLY selected levels)
        self.use_wfcoop = use_wfcoop
        self.wfcoop_idx = set(list(wfcoop_idx))
        if self.use_wfcoop:
            self.wfcoop = nn.ModuleDict()
            for i in range(len(in_channels)):
                self.wfcoop[str(i)] = WaveletFourierCoop(hidden_dim, act=act) if i in self.wfcoop_idx else nn.Identity()

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=enc_act
        )
        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers)
            for _ in range(len(use_encoder_idx))
        ])

        input_dim = hidden_dim if self.fuse_op == 'sum' else hidden_dim * 2

        Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1)
        SCDown_Conv = nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2))

        c1, c2, c3, c4, num_blocks = input_dim, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult)
        if version == 'dfine':
            Fuse_Block = RepNCSPELAN4(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act, csp_type=csp_type)
        elif version == 'deim':
            Fuse_Block = RepNCSPELAN5(c1=c1, c2=c2, c3=c3, c4=c4, n=num_blocks, act=act)
        else:
            Fuse_Block = CSPLayer(in_channels=c1, out_channels=c2, num_blocks=num_blocks, act=act,
                                  expansion=expansion, bottletype=VGGBlock)
            Lateral_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act)
            SCDown_Conv = ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(copy.deepcopy(Lateral_Conv))
            self.fpn_blocks.append(copy.deepcopy(Fuse_Block))

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(copy.deepcopy(SCDown_Conv))
            self.pan_blocks.append(copy.deepcopy(Fuse_Block))

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride,
                    self.eval_spatial_size[0] // stride,
                    self.hidden_dim,
                    self.pe_temperature
                )
                setattr(self, f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]
        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats: List[torch.Tensor]):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]

        # Apply W-F coop only on S3/S4
        if getattr(self, "use_wfcoop", False):
            for i in range(len(proj_feats)):
                proj_feats[i] = self.wfcoop[str(i)](proj_feats[i])

        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)

                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature
                    ).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory: torch.Tensor = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()

        # FPN
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            fused_feat = (upsample_feat + feat_low) if self.fuse_op == 'sum' else torch.concat([upsample_feat, feat_low], dim=1)
            inner_out = self.fpn_blocks[len(self.in_channels) - 1 - idx](fused_feat)
            inner_outs.insert(0, inner_out)

        # PAN
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            fused_feat = (downsample_feat + feat_high) if self.fuse_op == 'sum' else torch.concat([downsample_feat, feat_high], dim=1)
            out = self.pan_blocks[idx](fused_feat)
            outs.append(out)

        return outs
