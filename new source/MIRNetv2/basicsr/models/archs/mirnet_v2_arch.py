"""
MIRNetv2 Architecture for Low-Light Image Enhancement
Based on: "MIRNetv2: Learning Enhanced Representations for Illumination-Aware Image Enhancement"
https://github.com/swz30/MIRNetv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class LayerNorm(nn.Module):
    """Layer Normalization with channels_first support"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class MRB(nn.Module):
    """Multi-Receptive Field Module"""
    def __init__(self, in_channels, out_channels, n_feat=80, chan_factor=1.5, height=3, width=2):
        super().__init__()

        self.n_feat = n_feat
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Compress channels
        self.conv1 = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=1)

        # Global Context Block
        self.gca = GlobalContextBlock(n_feat, n_feat)

        # Spatial Attention
        self.spa = SpatialAttentionBlock(n_feat)

        # Channel Attention
        self.cha = ChannelAttentionBlock(n_feat, reduction=16)

        # Output projection
        self.conv2 = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1)

        # Residual connection
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.conv1(x)
        x = x + self.gca(x)
        x = x + self.spa(x)
        x = x + self.cha(x)
        x = self.conv2(x)
        return x + residual


class GlobalContextBlock(nn.Module):
    """Global Context Block for long-range dependencies"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention_pooling = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.LayerNorm([out_channels, 1, 1]) if hasattr(nn, 'LayerNorm') else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        # Attention pooling
        attn = self.attention_pooling(x)
        attn = F.softmax(attn.view(b, 1, -1), dim=-1).view(b, 1, h, w)
        # Global context
        ctx = torch.sum(x * attn, dim=(2, 3), keepdim=True)
        transform = self.transform(ctx)
        return x + transform


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention Block"""
    def __init__(self, channels):
        super().__init__()
        self.conv_spatial = nn.Conv2d(channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention map
        attn = self.conv_spatial(x)
        attn = self.sigmoid(attn)
        # Apply attention - broadcast across channels
        return x * attn


class ChannelAttentionBlock(nn.Module):
    """Channel Attention Block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class RRG(nn.Module):
    """Recursive Residual Group"""
    def __init__(self, n_feat, n_mrb, chan_factor=1.5):
        super().__init__()
        self.n_feat = n_feat
        self.n_mrb = n_mrb

        # Multiple MRB blocks in series
        self.mrbs = nn.ModuleList([
            MRB(n_feat, n_feat, n_feat=n_feat, chan_factor=chan_factor)
            for _ in range(n_mrb)
        ])

        # Fusion
        self.fusion = nn.Conv2d(n_feat * n_mrb, n_feat, kernel_size=1)

    def forward(self, x):
        residuals = []
        for mrb in self.mrbs:
            x = mrb(x)
            residuals.append(x)
        x = torch.cat(residuals, dim=1)
        x = self.fusion(x)
        return x


class DownsampleBlock(nn.Module):
    """Downsample block for feature extraction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels) if out_channels <= 256 else nn.Identity()

    def forward(self, x):
        return F.relu(self.norm(self.conv(x)))


class UpsampleBlock(nn.Module):
    """Upsample block for feature reconstruction"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.norm = nn.BatchNorm2d(out_channels) if out_channels <= 256 else nn.Identity()

    def forward(self, x):
        return F.relu(self.norm(self.pixel_shuffle(self.conv(x))))


class MIRNet_v2(nn.Module):
    """
    MIRNetv2 for Low-Light Image Enhancement
    """
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        bias=False,
        scale=1,
        task=None
    ):
        super().__init__()

        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.n_RRG = n_RRG
        self.n_MRB = n_MRB
        self.scale = scale

        # Initial feature extraction
        self.conv_first = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        # Hierarchical Feature Extraction
        self.RRG_blocks = nn.ModuleList()
        for i in range(n_RRG):
            self.RRG_blocks.append(
                RRG(n_feat, n_MRB, chan_factor)
            )

        # Cross-scale fusion
        self.fusion = nn.Conv2d(n_feat * n_RRG, n_feat, kernel_size=1, bias=bias)

        # Output reconstruction
        self.conv_last = nn.Conv2d(n_feat, out_channels * (scale ** 2), kernel_size=3, padding=1, bias=bias)

        if scale > 1:
            self.pixel_shuffle = nn.PixelShuffle(scale)
        else:
            self.pixel_shuffle = nn.Identity()

        # Activation
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        # Initial feature extraction
        feat = self.conv_first(x)
        feat = self.lrelu(feat)

        # Hierarchical feature extraction with RRG blocks
        rrg_outs = []
        for rrg in self.RRG_blocks:
            feat = rrg(feat)
            rrg_outs.append(feat)

        # Cross-scale fusion
        feat = torch.cat(rrg_outs, dim=1)
        feat = self.fusion(feat)

        # Output reconstruction
        out = self.conv_last(feat)
        out = self.pixel_shuffle(out)

        # Clamp output to valid range
        out = torch.clamp(out, 0, 1)

        return out


def build_mirnet_v2(**kwargs):
    """Build MIRNetv2 model"""
    return MIRNet_v2(**kwargs)


# For backward compatibility - alias to the MIRNet_v2 class
# The actual class is defined above
MIRNetV2 = MIRNet_v2
