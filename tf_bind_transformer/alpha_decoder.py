import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from tf_bind_transformer.alpha_encoder import ConvBlock


class UpResBlock(nn.Module):
    def __init__(self, dim, skip_dim):
        super().__init__()
        self.conv1 = ConvBlock(dim, skip_dim)
        self.conv2 = ConvBlock(skip_dim, skip_dim, kernel_size=1)
        self.conv3 = ConvBlock(skip_dim, skip_dim)

        self.residual_scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, skip):
        out = self.conv1(x) + x[:, :, : skip.shape[-1]]
        out = F.interpolate(
            out.transpose(1, 2), scale_factor=2, mode="nearest"
        ).transpose(1, 2)
        out = out * self.residual_scale

        skip_processed = self.conv2(skip)
        out = out + skip_processed

        out = out + self.conv3(out)
        return out
