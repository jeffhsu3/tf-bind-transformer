# %%
import unittest
import torch
import pyBigWig
import numpy as np
import polars as pl
from pathlib import Path
import yaml
import rich
from einops import rearrange
import wandb

from tf_bind_transformer.data_bigwig2 import BigWigTracksOnlyDataset, get_bigwig_tracks_dataloader
from tf_bind_transformer.data import FactorProteinDataset # Assuming this is needed for BigWigDataset
from tf_bind_transformer.mini_alphgenome import create_mini_alphagenome, multinomial_loss, TFChIPSeqOutputHead 
from tf_bind_transformer.models.deepsea import ImprovedDeepSEA

# Configuration flag for WandB logging
USE_WANDB = False # Set to False to disable WandB logging

with open('/home/jeff/iv4/ref/ecoli/ecoli.yaml') as f:
    config = yaml.safe_load(f)

bw_seq_len = 4096
nlayers = 4
# Calculate downsampling amount
# :TODO make sure 
trunk_seqlen = bw_seq_len/(2**(nlayers-1))
down_sample_factor = int(bw_seq_len / trunk_seqlen)

print(f"Trunk seqlen = {trunk_seqlen}")
print(f"Downsample Factor = {down_sample_factor}")

dataset_train = BigWigTracksOnlyDataset(
    #bigwig_folder='tfactor_faa/bigwig/',
    bigwig_folder='./bws/',
    enformer_loci_path=f'cv_splits_{bw_seq_len}/fold_1_train_loci.bed',
    fasta_file=config['genome_fasta'],
    ref='ASM584v2',
    annot_file='bigwig_track_means.csv',
    downsample_factor=down_sample_factor,
    target_length=trunk_seqlen,
)
dataset_test = BigWigTracksOnlyDataset(
    bigwig_folder='./bws/',
    enformer_loci_path=f'cv_splits_{bw_seq_len}/fold_1_test_loci.bed',
    fasta_file=config['genome_fasta'],
    annot_file='bigwig_track_means.csv',
    ref='ASM584v2',
    downsample_factor=down_sample_factor,
    target_length=trunk_seqlen
)

dataloader = get_bigwig_tracks_dataloader(dataset_train, batch_size=4, cycle_iter=False)
# Add an n_lyaer 
model = create_mini_alphagenome(base_dim=32, layers=nlayers, num_targets=dataset_train.ntargets)
BATCH_SIZE=2
# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#model = ImprovedDeepSEA(n_targets=dataset.ntargets)

model = model.to(device)
optimizer = torch.optim.Adam(list(model.parameters()))

# Track means for scaling (would normally be calculated from dataset)
# Initialize wandb
if USE_WANDB:
    wandb.init(
        project="tf-bind-transformer",
        name="bigwig-test-run",
        config={
            "model": "mini_alphagenome",
            "base_dim": 32,
            "layers": 5,
            "batch_size": 4,
            "learning_rate": 1e-4,
            "num_epochs": 3,
            "num_targets": dataset_train.ntargets,
            "device": str(device),
            "downsample_factor": 16,
        }
    )

track_means = torch.ones(dataset_train.ntargets) * 2.0

print(f"Number of targets: {dataset_train.ntargets}")
model_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {model_params:,}")

# Log model parameters to wandb
if USE_WANDB:
    wandb.log({"model_parameters": model_params})

mloss = torch.nn.MSELoss()
# Training loop
num_epochs = 3
step = 0

# Create test dataloader
test_dataloader = get_bigwig_tracks_dataloader(dataset_test, batch_size=BATCH_SIZE, cycle_iter=False)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (sequences, targets) in enumerate(dataloader):
        # Move data to GPU
        sequences = sequences.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        
        # Forward pass through model
        predictions = model(sequences)
        
        # Generate predictions using output head
        #predictions = output_head(embeddings)
        # Calculate multinomial loss
        #loss = multinomial_loss(predictions, targets, multinomial_resolution=1024)
        loss = mloss(predictions, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        step += 1
        
        if USE_WANDB:
            wandb.log({
                "train/loss": loss.item(),
                "train/epoch": epoch + 1,
                "train/batch": batch_idx + 1,
                "train/step": step,
            })
        
        if batch_idx == 0:  # Print shapes for first batch
            print(f"\nEpoch {epoch + 1}, Batch {batch_idx + 1}:")
            print(f"  Sequences shape: {sequences.shape}")
            print(f"  Targets shape: {targets.shape}")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Device: {sequences.device}")
            
            # Log additional metrics for first batch
            if USE_WANDB:
                wandb.log({
                    "debug/sequences_shape": list(sequences.shape),
                    "debug/targets_shape": list(targets.shape),
                    "debug/predictions_shape": list(predictions.shape),
                    "debug/predictions_mean": predictions.mean().item(),
                    "debug/predictions_std": predictions.std().item(),
                    "debug/targets_mean": targets.mean().item(),
                    "debug/targets_std": targets.std().item(),
                })
        
        print(f"  Loss: {loss.item():.4f}")
    
    # Calculate average training loss safely
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Training Loss: {avg_loss:.4f}")
    else:
        avg_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, No training batches processed")
    
    # Log epoch metrics
    if USE_WANDB:
        wandb.log({
            "train/epoch_loss": avg_loss,
            "train/epoch_num": epoch + 1,
        })
    
    # Test phase
    model.eval()
    test_total_loss = 0
    test_num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(test_dataloader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions = model(sequences)
            loss = mloss(predictions, targets)
            
            test_total_loss += loss.item()
            test_num_batches += 1
    
    # Calculate average test loss safely
    if test_num_batches > 0:
        avg_test_loss = test_total_loss / test_num_batches
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Test Loss: {avg_test_loss:.4f}")
    else:
        avg_test_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}, No test batches processed")
    
    # Log test metrics
    if USE_WANDB:
        wandb.log({
            "test/epoch_loss": avg_test_loss,
            "test/epoch_num": epoch + 1,
        })

print("\nTraining completed!")
if USE_WANDB:
    wandb.finish()

# %%
