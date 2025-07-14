""" A modified Deepsea model for track height prediction
"""
import torch
import torch.nn as nn


class ImprovedDeepSEA(nn.Module):
    def __init__(self, n_targets):
        """
        A refactored DeepSEA model with modern best practices.
        Note: sequence_length is no longer needed in the constructor.
        """
        super().__init__()
        
        # --- Convolutional Block 1 ---
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(4, 320, kernel_size=8, padding='same'),
            nn.BatchNorm1d(320),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4), # Pooling handles dimensionality reduction
            nn.Dropout(p=0.2)
        )
        
        # --- Convolutional Block 2 ---
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(320, 480, kernel_size=8, padding='same'),
            nn.BatchNorm1d(480),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(p=0.2)
        )
        
        # --- Convolutional Block 3 ---
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(480, 960, kernel_size=8, padding='same'),
            nn.BatchNorm1d(960),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        
        # --- Adaptive Pooling and Classifier ---
        # This replaces the brittle manual shape calculation
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()

        self.regressor = nn.Sequential(
            nn.Linear(960, n_targets),
            nn.ReLU(inplace=True),
            nn.Linear(n_targets, n_targets)
        )


    def forward(self, x):
        # Explicit forward pass
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Automatically handle shape for the linear layer
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        return self.regressor(x)