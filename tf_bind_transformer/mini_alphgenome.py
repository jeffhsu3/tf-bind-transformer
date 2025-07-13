import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import einsum

from tf_bind_transformer.alpha_encoder import ConvBlock, SequenceEncoder, RMSBatchNorm1d


class TFChIPSeqOutputHead(nn.Module):
    """
    Output head for TF ChIP-seq tracks as described in Alpha Genome paper.
    Applies softplus activation and learnable per-track scaling.
    :TODO double-check this
    """

    def __init__(self, input_dim: int, num_tracks: int):
        super().__init__()
        self.num_tracks = num_tracks
        self.linear = nn.Linear(input_dim, num_tracks)
        # Initialize scale parameter to 0.0 as specified in paper
        self.scale = nn.Parameter(torch.zeros(num_tracks))

    def forward(self, embeddings):
        embeddings = rearrange(embeddings, 'b c s -> b s c' )
        x = self.linear(embeddings)
        #x = rearrange(x, 'b c s -> b s c' )
        return F.softplus(x) * F.softplus(self.scale)


def targets_scaling(targets, track_means, apply_squashing=False):
    """
    Scale targets before loss calculation for numerical stability. This should be down on the data itself beforehand

    Args:
        targets: Array of shape [seq_len, num_tracks]
        track_means: Array of shape [num_tracks] with pre-calculated means
        apply_squashing: bool, whether to apply power transformation (RNA-seq only)

    Returns:
        Scaled targets
    """
    # Normalize by track means
    targets = targets / track_means

    # Apply power transformation for RNA-seq tracks only
    if apply_squashing:
        targets = targets**0.75

    # Apply smooth clipping for high values
    return torch.where(targets > 10.0, 2 * torch.sqrt(targets * 10.0) - 10.0, targets)


def predictions_scaling(x, track_means, apply_squashing=False):
    """
    Inverse scaling for predictions to match original experimental data scale.

    Args:
        x: Predictions of shape [seq_len, num_tracks]
        track_means: Array of shape [num_tracks] with pre-calculated means
        apply_squashing: bool, whether to reverse power transformation

    Returns:
        Inverse scaled predictions
    """
    # Reverse smooth clipping
    x = torch.where(x > 10.0, (x + 10.0) ** 2 / (4 * 10.0), x)

    # Reverse power transformation for RNA-seq tracks
    if apply_squashing:
        x = x ** (1.0 / 0.75)
    # Reverse normalization
    return x * track_means


def multinomial_loss(predictions, targets, multinomial_resolution=1024):
    """
    Calculate multinomial loss combining Poisson and Multinomial NLL terms.

    Args:
        predictions: Tensor of shape [batch_size, seq_len, num_tracks]
        targets: Tensor of shape [batch_size, seq_len, num_tracks]
        multinomial_resolution: int, resolution for multinomial calculation (1024 for 128bp)

    Returns:
        Loss tensor
    """
    # Reshape to segments of multinomial_resolution
    seq_len, num_tracks = predictions.shape
    num_segments = seq_len // multinomial_resolution

    # Reshape to [batch_size * num_segments, multinomial_resolution, num_tracks]
    x = predictions[: num_segments * multinomial_resolution, :].reshape(
        multinomial_resolution, num_tracks
    )
    targets_reshaped = targets[: num_segments * multinomial_resolution, :].reshape(
        multinomial_resolution, num_tracks
    )

    # Sum over bins within each segment
    sum_pred = torch.sum(x, dim=1, keepdim=True)  # [batch*segments, 1, tracks]
    sum_target = torch.sum(
        targets_reshaped, dim=1, keepdim=True
    )  # [batch*segments, 1, tracks]

    # Poisson NLL term: sum(pred) - sum(target) * log(sum(pred) + eps)
    poisson_loss = torch.sum(sum_pred - sum_target * torch.log(sum_pred + 1e-7))

    # Multinomial probability distribution
    multinomial_prob = x / (sum_pred + 1e-7)

    # Multinomial NLL term: -sum(target * log(prob + eps))
    positional_loss = torch.sum(-targets_reshaped * torch.log(multinomial_prob + 1e-7))

    # Combine losses with specified weights
    return poisson_loss / multinomial_resolution + 5.0 * positional_loss


class TFChIPSeqLoss(nn.Module):
    """
    Complete loss function for TF ChIP-seq tracks following Alpha Genome paper.
    """

    def __init__(self, track_means, multinomial_resolution=1024):
        super().__init__()
        self.track_means = track_means
        self.multinomial_resolution = multinomial_resolution

    def forward(self, predictions, targets):
        """
        Calculate loss for TF ChIP-seq tracks.

        Args:
            predictions: Raw model predictions [batch_size, seq_len, num_tracks]
            targets: Ground truth targets [batch_size, seq_len, num_tracks]

        Returns:
            Loss value
        """
        # Scale targets (no squashing for ChIP-seq)
        scaled_targets = targets_scaling(
            targets, self.track_means, apply_squashing=False
        )

        # Calculate multinomial loss
        loss = multinomial_loss(
            predictions, scaled_targets, self.multinomial_resolution
        )

        return loss


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position=1024):
        super().__init__()
        self.dim = dim
        self.max_position = max_position

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, positions=None):
        if positions is None:
            positions = torch.arange(x.shape[1], device=x.device).float()

        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)

        cos_emb = emb.cos()
        sin_emb = emb.sin()

        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos_emb + rotated * sin_emb


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, heads * dim_head, bias=False)
        self.to_k = nn.Linear(dim, dim_head, bias=False)
        self.to_v = nn.Linear(dim, dim_head, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)

        self.rope = RotaryEmbedding(dim_head)

    def forward(self, x, attention_bias=None):
        batch, seq_len, _ = x.shape

        x = self.norm(x)

        q = self.to_q(x).view(batch, seq_len, self.heads, self.dim_head)
        k = self.to_k(x).view(batch, seq_len, 1, self.dim_head)
        v = self.to_v(x).view(batch, seq_len, 1, self.dim_head)

        q = self.rope(q)
        k = self.rope(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_bias is not None:
            attn = attn + attention_bias

        attn = torch.tanh(attn / 5.0) * 5.0
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.to_out(out)


class MLPBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.mlp(x)


class SimplePairwiseBlock(nn.Module):
    def __init__(self, seq_dim, pair_dim=32):
        super().__init__()
        self.seq_dim = seq_dim
        self.pair_dim = pair_dim

        self.to_q = nn.Linear(seq_dim, pair_dim, bias=False)
        self.to_k = nn.Linear(seq_dim, pair_dim, bias=False)
        self.to_v = nn.Linear(seq_dim, pair_dim, bias=False)

        self.proj = nn.Linear(pair_dim, pair_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pair_dim, pair_dim * 2),
            nn.ReLU(),
            nn.Linear(pair_dim * 2, pair_dim),
        )

    def forward(self, x, pair_state=None):
        batch, seq_len, _ = x.shape

        # Downsample sequence for pairwise computation
        x_pooled = F.avg_pool1d(x.transpose(1, 2), kernel_size=8).transpose(1, 2)

        q = self.to_q(x_pooled)
        k = self.to_k(x_pooled)
        v = self.to_v(x_pooled)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.pair_dim)
        pair_update = torch.matmul(attn, v)

        pair_proj = self.proj(pair_update)

        if pair_state is None:
            pair_state = pair_proj
        else:
            pair_state = pair_state + pair_proj

        pair_state = pair_state + self.mlp(pair_state)

        return pair_state


class TransformerTower(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, pair_dim=32):
        super().__init__()
        self.attention = MultiHeadAttention(dim, heads, dim_head)
        self.mlp = MLPBlock(dim)
        self.pairwise = SimplePairwiseBlock(dim, pair_dim)
        self.attention_bias_proj = nn.Linear(pair_dim, heads, bias=False)

    def forward(self, x):
        pair_state = self.pairwise(x)

        # Create attention bias from pairwise state
        attention_bias = self.attention_bias_proj(pair_state)
        attention_bias = attention_bias.permute(0, 3, 1, 2)

        # Repeat to match sequence length if needed
        if attention_bias.shape[-1] != x.shape[1]:
            repeat_factor = x.shape[1] // attention_bias.shape[-1]
            attention_bias = attention_bias.repeat_interleave(repeat_factor, dim=-1)
            attention_bias = attention_bias.repeat_interleave(repeat_factor, dim=-2)

        attn_out = self.attention(x, attention_bias)
        x = x + attn_out

        mlp_out = self.mlp(x)
        x = x + mlp_out

        return x, pair_state


class UpResBlock(nn.Module):
    def __init__(self, dim: int, unet_skip: int):
        super().__init__()
        self.dim = dim
        # self.out_dim = out_dim = unet_skip.shape[2]
        self.out_dim = unet_skip

        self.conv1 = ConvBlock(dim, self.out_dim)
        self.skip_conv = ConvBlock(self.out_dim, self.out_dim, kernel_size=1)
        self.conv2 = ConvBlock(self.out_dim, self.out_dim)
        self.residual_scale = nn.Parameter(torch.tensor(0.9))

    def forward(self, x, skip):
        out = self.conv1(x) + x[:, : self.out_dim, :]
        out = torch.repeat_interleave(out, 2, dim=2) * self.residual_scale
        out += self.skip_conv(skip)
        return out + self.conv2(out)


class SequenceDecoder(nn.Module):
    def __init__(
        self,
        base_dim=128,
        feat_growth: int = 4,
        layers: int = 4,
        intermediate_dims=None,
    ):
        super().__init__()
        # Reverse the encoder dimensions
        self.layers = layers
        dims = [base_dim + (self.layers - 1) * feat_growth] + [
            base_dim + i * feat_growth for i in reversed(range(layers))
        ][:-1]
        skip_dims = [base_dim + (i) * feat_growth for i in reversed(range(layers))]
        skip_dims[-1] = base_dim
        # Maybe just pass in a dict with the intermediate dims?

        self.up_blocks = nn.ModuleList(
            [UpResBlock(dims[i], skip_dims[i]) for i in range(layers)]
        )

    def forward(self, x, intermediates):
        bin_sizes = list(reversed([2 ** (i + 1) for i in range(self.layers - 1)]))
        for i, (up_block, bin_size) in enumerate(zip(self.up_blocks, bin_sizes)):
            skip = intermediates[f"bin_size_{bin_size}"]
            #print(f"Skip shape {skip.shape}")
            x = up_block(x, skip)

        return x


class EColiOutputHead(nn.Module):
    def __init__(self, dim, skip_x=False):
        super().__init__()
        #self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = RMSBatchNorm1d(dim*2)
        self.activation = nn.GELU()

    def forward(self, x, skip_x: torch.Tensor | None =None):
        x = rearrange(x, 'b c s -> b s c')
        x = self.linear(x)
        x = rearrange(x, 'b s c -> b c s')
        #if skip_x is not None:
        #    skip_x = 
        # No don't learn organism specific embedding as of yet
        return self.activation(self.norm(x))


class MiniAlphaGenome(nn.Module):
    def __init__(self, 
    base_dim=128, 
    heads=4, 
    dim_head=32, 
    pair_dim=32, 
    num_tracks=1,
    n_unet_layers=4,
    feat_growth=4,
    decoder = False,
    ):
        super().__init__()

        self.encoder = SequenceEncoder(base_dim, layers=n_unet_layers)
        tdim = base_dim + (n_unet_layers-1) * feat_growth
        # self.transformer = TransformerTower(tdim, heads, dim_head, pair_dim)
        if decoder:
            self.decoder = SequenceDecoder(base_dim)
        else: 
            pass
        self.embeddings_low = EColiOutputHead(tdim)
        self.chipseq_head = TFChIPSeqOutputHead(tdim*2, num_tracks)

    def forward(self, x, decoder=False):
        trunk, intermediates = self.encoder(x)
        # :TODO need to add an organism embedding here
        # :TODO fix the transformer tower layers
        # trunk, pair_state = self.transformer(trunk)

        if decoder:
            decoded = self.decoder(trunk, intermediates)
        else: 
            pass
        #print(f'trunk shape {trunk.shape}')
        # Low resolution embeddings
        embeddings_low = self.embeddings_low(trunk)
        # return output, pair_state
        out = self.chipseq_head(embeddings_low)
        return out 



def create_mini_alphagenome(
    sequence_length=1024,
    base_dim=128,
    heads=4,
    dim_head=32,
    pair_dim=32,
    num_targets=1,
    layers=4,
    transformer_tower=False,
):
    if transformer_tower:
        raise NotImplementedError
    model = MiniAlphaGenome(
        base_dim=base_dim,
        heads=heads,
        dim_head=dim_head,
        pair_dim=pair_dim,
        n_unet_layers=layers,
        num_tracks=num_targets,
    )
    return model


if __name__ == "__main__":
    # Create model
    model = create_mini_alphagenome()

    # Test with sample data
    batch_size = 2
    seq_len = 1024

    # Random DNA sequence (one-hot encoded)
    x = torch.randn(batch_size, seq_len, 4)
    x = F.softmax(x, dim=-1)  # Convert to proper one-hot probabilities

    print("Testing MiniAlphaGenome model...")
    print(f"Input shape: {x.shape}")

    # Forward pass
    output, pair_state = model(x)

    print(f"Output shape: {output.shape}")
    # print(f"Pair state shape: {pair_state.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Model Encoder parameters: {sum(p.numel() for p in model.encoder.parameters()):,}"
    )
    print("\nModel created successfully!")

    # Example usage of TF ChIP-seq loss function
    print("\n" + "=" * 50)
    print("Testing TF ChIP-seq Loss Function")
    print("=" * 50)

    # Create sample data for loss function testing
    batch_size = 2
    seq_len = 8192  # 128bp resolution sequence length
    num_tracks = 4

    # Create TF ChIP-seq output head
    input_dim = 128  # Assuming 128-dimensional embeddings
    tf_head = TFChIPSeqOutputHead(input_dim, num_tracks)

    # Sample embeddings from the model
    embeddings = torch.randn(batch_size, seq_len, input_dim)

    # Generate predictions using the output head
    predictions = tf_head(embeddings)
    print(f"Predictions shape: {predictions.shape}")
    print(
        f"Predictions min/max: {predictions.min().item():.4f} / {predictions.max().item():.4f}"
    )

    # Create sample targets (ChIP-seq read counts)
    targets = torch.poisson(torch.ones(batch_size, seq_len, num_tracks) * 2.0)
    print(f"Targets shape: {targets.shape}")
    print(f"Targets min/max: {targets.min().item():.4f} / {targets.max().item():.4f}")

    # Pre-calculated track means (would be calculated from entire dataset)
    track_means = torch.tensor([1.5, 2.0, 1.8, 2.2])

    # Create loss function
    loss_fn = TFChIPSeqLoss(track_means, multinomial_resolution=1024)

    # Calculate loss
    loss = loss_fn(predictions, targets)
    print(f"TF ChIP-seq loss: {loss.item():.4f}")

    # Test scaling functions
    print("\nTesting scaling functions...")
    scaled_targets = targets_scaling(targets, track_means, apply_squashing=False)
    print(f"Scaled targets shape: {scaled_targets.shape}")

    # Test inverse scaling
    rescaled_predictions = predictions_scaling(
        predictions, track_means, apply_squashing=False
    )
    print(f"Rescaled predictions shape: {rescaled_predictions.shape}")
    print("\nTF ChIP-seq loss function implementation complete!")
