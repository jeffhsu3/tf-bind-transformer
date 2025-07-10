import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class StandardizedConv1d(nn.Conv1d):
    """ Standardized 1D convolution

    The weights of the convolution are standardized befor the forward pass.
    """
    def forward(self, input):
        weight = self.weight
        weight_mean = weight.mean(dim=(1, 2), keepdim=True)
        weight = weight - weight_mean
        std = weight.std(dim=(1, 2), keepdim=True) + 1e-5
        weight = weight / std
        return F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class RMSBatchNorm1d(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-6, momentum: float = 0.1):
        """
        Root Mean Square Batch Normalization for 1D data w/o mean subtraction.

        Args:
            num_features: The number of channels in the input tensor.
            eps: A small value to prevent division by zero.
            momentum: The momentum for updating the running variance.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_var", torch.ones(num_features))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (got {x.dim()}D input)")
        if self.training:
            var = torch.mean(x.pow(2), dim=[0, 2])
            self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
            current_var = var
        else:
            current_var = self.running_var

        inv_std = 1.0 / torch.sqrt(current_var.view(1, self.num_features, 1) + self.eps)
        return x * inv_std * self.weight.view(
            1, self.num_features, 1
        ) + self.bias.view(1, self.num_features, 1)


class ConvBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, kernel_size: int = 5):
        super().__init__()
        self.rms_batch_norm = RMSBatchNorm1d(num_features=dim)
        self.activation = nn.GELU()

        if kernel_size == 1:
            self.conv = StandardizedConv1d(dim, out_dim, 1, padding=0)
        else:
            self.conv = StandardizedConv1d(dim, out_dim, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = self.rms_batch_norm(x)
        x = self.activation(x)
        x = self.conv(
            x,
        )
        return x


class DNAEmbedder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        # Why is the padding 7
        self.initial_conv = nn.Conv1d(4, dim, 15, padding=7)
        self.conv_block = ConvBlock(dim, dim)

    def forward(self, x):
        #x = x.transpose(1, 2)  # (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = rearrange(x, 'b s c -> b c s')
        out = self.initial_conv(x)  # Conv1d expects (batch, channels, seq_len)
        return out + self.conv_block(out)  # Keep in (batch, channels, seq_len) format


class DownResBlock(nn.Module):
    def __init__(self, dim: int, feature_add: int=4):
        super().__init__()
        self.out_dim = dim + feature_add
        self.conv1 = ConvBlock(dim, self.out_dim)
        self.conv2 = ConvBlock(self.out_dim, self.out_dim)

    def forward(self, x):
        out = self.conv1(x)
        # Pad input x to match the output channels from conv1
        # F.pad for channel dimension: (left, right, top, bottom, front, back)
        # For (batch, channels, seq_len), we pad the channel dimension (dim=1)
        padded_x = F.pad(x, (0, 0, 0, out.shape[1] - x.shape[1]))
        out = out + padded_x
        return out + self.conv2(out)


class SequenceEncoder(nn.Module):
    """
    layers: int
        Number of downres blocks to use.
    """
    def __init__(self, base_dim=8, feat_growth: int=4, layers: int=4):
        super().__init__()
        self.dna_embedder = DNAEmbedder(base_dim)
        dims = [base_dim + i*feat_growth for i in range(layers)]
        
        self.down_blocks = nn.ModuleList(
            [DownResBlock(dims[i], 4) for i in range(layers-1)]
        )
        self.max_pools = nn.ModuleList([nn.MaxPool1d(2) for _ in range(3)])

    def forward(self, x):
        intermediates = {}
        x = self.dna_embedder(x)
        intermediates["bin_size_1"] = x
        for i, (down_block, max_pool) in enumerate(
            zip(self.down_blocks, self.max_pools)
        ):
            x = down_block(x)
            bin_size = 2 ** (i + 1)
            intermediates[f"bin_size_{bin_size}"] = x
            x = max_pool(x)

        return x, intermediates





if __name__ == "__main__":
    # Create model
    model = SequenceEncoder(8)
    # Test with sample data
    batch_size = 2
    seq_len = 1024

    # Random DNA sequence (one-hot encoded) - keep in (batch, seq_len, channels) format
    x = torch.randn(batch_size, seq_len, 4)
    x = F.softmax(x, dim=-1)  # Convert to proper one-hot probabilities

    print("Testing SequenceEncoder model...")
    print(f"Input shape: {x.shape}")

    # Forward pass
    output, intermediates = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Intermediates keys: {list(intermediates.keys())}")
    for key, tensor in intermediates.items():
        print(f"  {key}: {tensor.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test with different sequence length
    print("\nTesting with different sequence length...")
    x_short = torch.randn(1, 512, 4)
    x_short = F.softmax(x_short, dim=-1)

    output_short, intermediates_short = model(x_short)
    print(f"Short input shape: {x_short.shape}")
    print(f"Short output shape: {output_short.shape}")
    print("\nModel created successfully!")
