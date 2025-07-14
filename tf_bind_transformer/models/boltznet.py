# Implementation of BoltzNet
# Two-stage CNN architecture for modeling TF binding affinity and ChIP-Seq coverage
# Based on: BoltzNet: A physics-based model for predicting transcription factor binding
#@article{lally2025predictive,
#  title={Predictive biophysical neural network modeling of a compendium of in vivo transcription factor DNA binding profiles for Escherichia coli},
#  author={Lally, Patrick and G{\'o}mez-Romero, Laura and Tierrafr{\'\i}a, V{\'\i}ctor H and Aquino, Patricia and Rioualen, Claire and Zhang, Xiaoman and Kim, Sunyoung and Baniulyte, Gabriele and Plitnick, Jonathan and Smith, Carol and others},
#  journal={Nature Communications},
#  volume={16},
#  number={1},
#  pages={4255},
#  year={2025},
#  publisher={Nature Publishing Group UK London}
#}

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BoltzNet(nn.Module):
    """
    BoltzNet: A two-stage CNN architecture that mirrors biophysical TF binding.
    
    Stage 1: Thermodynamic binding affinity modeling using convolution + exponential activation
    Stage 2: ChIP-Seq coverage prediction using fully connected network
    
    Args:
        ntargets (int): Number of ChIP-Seq experiments to predict coverage for
        kernel_width (int): Width of convolution kernel (default: 25bp as per paper)
        sequence_length (int): Length of input DNA sequences (default: 101bp as per paper)
    """
    
    def __init__(self, ntargets, kernel_width=25, sequence_length=101, 
                 nn_hidden_nodes=64, nn_layers=2, l1_reg=0.0):
        super(BoltzNet, self).__init__()
        
        self.ntargets = ntargets
        self.kernel_width = kernel_width
        self.sequence_length = sequence_length
        self.l1_reg = l1_reg
        
        # Stage 1: Thermodynamic binding affinity component
        # Single convolution kernel to model binding energy matrix
        # Input: 4 channels (A,C,G,T), Output: 1 channel (binding affinity)
        self.affinity_conv = nn.Conv1d(
            in_channels=4, 
            out_channels=1, 
            kernel_size=kernel_width,
            bias=True,
            padding=0
        )
        
        # Calculate output length after convolution
        conv_output_length = sequence_length - kernel_width + 1
        
        # Stage 2: ChIP-Seq coverage prediction network
        # Fully connected network that maps affinity score to coverage values
        # Universal function approximator as described in paper
        layers = []
        
        # Input layer
        layers.append(nn.Linear(1, nn_hidden_nodes))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(nn_layers - 1):
            layers.append(nn.Linear(nn_hidden_nodes, nn_hidden_nodes))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(nn_hidden_nodes, ntargets))
        
        self.coverage_net = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass through BoltzNet.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, 4, sequence_length)
                             One-hot encoded with channels for A, C, G, T
        
        Returns:
            torch.Tensor: Predicted ChIP-Seq coverage of shape (batch_size, ntargets)
        """
        batch_size = x.size(0)
        x = rearrange(x, 'b s c -> b c s')
        
        # Stage 1: Thermodynamic binding affinity modeling
        # Apply convolution to get binding energies at each position
        affinity_scores = self.affinity_conv(x)  # Shape: (batch_size, 1, conv_output_length)
        
        # Apply exponential activation to model Boltzmann distribution
        # This corresponds to exp(Δε) in the thermodynamic model
        exp_affinities = torch.exp(affinity_scores)  # Shape: (batch_size, 1, conv_output_length)
        
        # Average exponential affinities across all positions (matches original implementation)
        # Original paper uses average pooling which works better than sum or max pooling
        total_affinity = torch.mean(exp_affinities, dim=2)  # Shape: (batch_size, 1)
        
        coverage = self.coverage_net(total_affinity)  # Shape: (batch_size, ntargets)
        
        return coverage
    
    def get_affinity_scores(self, x):
        """
        Extract just the affinity scores without coverage prediction.
        Useful for analysis and visualization.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, 4, sequence_length)
        
        Returns:
            torch.Tensor: Total affinity scores of shape (batch_size, 1)
        """
        with torch.no_grad():
            x = rearrange(x, 'b s c -> b c s')
            affinity_scores = self.affinity_conv(x)
            exp_affinities = torch.exp(affinity_scores)
            total_affinity = torch.mean(exp_affinities, dim=2)
            return total_affinity
    
    def get_position_affinities(self, x):
        """
        Get position-wise affinity scores for visualization.
        
        Args:
            x (torch.Tensor): Input DNA sequences of shape (batch_size, 4, sequence_length)
        
        Returns:
            torch.Tensor: Position-wise exp(affinity) scores of shape (batch_size, 1, conv_output_length)
        """
        with torch.no_grad():
            x = rearrange(x, 'b s c -> b c s')
            affinity_scores = self.affinity_conv(x)
            exp_affinities = torch.exp(affinity_scores)
            return exp_affinities
    
    def l1_loss(self):
        """
        Calculate L1 regularization loss for all linear layers.
        
        Returns:
            torch.Tensor: L1 regularization loss
        """
        l1_loss = 0.0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                l1_loss += torch.sum(torch.abs(module.weight))
                if module.bias is not None:
                    l1_loss += torch.sum(torch.abs(module.bias))
        return self.l1_reg * l1_loss