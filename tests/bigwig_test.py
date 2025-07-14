import torch
import torch.nn.functional as F
import pyBigWig
import numpy as np
from pathlib import Path
import yaml
import rich
from einops import rearrange
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
import math
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

from tf_bind_transformer.data_bigwig2 import BigWigTracksOnlyDataset, get_bigwig_tracks_dataloader
from tf_bind_transformer.data import FactorProteinDataset # Assuming this is needed for BigWigDataset
from tf_bind_transformer.mini_alphgenome import create_mini_alphagenome, multinomial_loss, TFChIPSeqOutputHead 
from tf_bind_transformer.models.deepsea import ImprovedDeepSEA
from tf_bind_transformer.models.boltznet import BoltzNet

# Training Configuration
CONFIG = {
    'use_wandb': False,
    'model': 'BOLTZNET',  # 'ALPHAGENOME' or 'BOLTZNET'
    'batch_size': 32,  # Increased for better gradient estimates
    'learning_rate': 1e-3,  # Higher initial LR with scheduling
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 10,  # Early stopping patience
    'grad_clip_norm': 1.0,  # Gradient clipping
    'warmup_epochs': 5,
    'use_mixed_precision': True,
    'save_checkpoints': True,
    'checkpoint_dir': './checkpoints/',
    'augment_data': True,  # Reverse complement augmentation
}

USE_WANDB = CONFIG['use_wandb']
MODEL = CONFIG['model']

with open('/home/jeff/iv4/ref/ecoli/ecoli.yaml') as f:
    config = yaml.safe_load(f)

bw_seq_len = 128

if MODEL == 'ALPHAGENOME':
    nlayers = 4
    trunk_seqlen = bw_seq_len/(2**(nlayers-1))
    down_sample_factor = int(bw_seq_len / trunk_seqlen)
elif MODEL == 'BOLTZNET':
    nlayers= 1
    trunk_seqlen = 1
    down_sample_factor = 128

print(f"Trunk seqlen = {trunk_seqlen}")
print(f"Downsample Factor = {down_sample_factor}")

# Create datasets with proper train/val/test split
dataset_train = BigWigTracksOnlyDataset(
    bigwig_folder='./bws/',
    enformer_loci_path=f'cv_splits_{bw_seq_len}/fold_1_train_loci.bed',
    fasta_file=config['genome_fasta'],
    ref='ASM584v2',
    annot_file='bigwig_track_means.csv',
    downsample_factor=down_sample_factor,
    target_length=trunk_seqlen,
)

# Use separate validation set (not test set during training)
dataset_val = BigWigTracksOnlyDataset(
    bigwig_folder='./bws/',
    enformer_loci_path=f'cv_splits_{bw_seq_len}/fold_1_val_loci.bed',  # TODO: Create proper val split
    fasta_file=config['genome_fasta'],
    annot_file='bigwig_track_means.csv',
    ref='ASM584v2',
    downsample_factor=down_sample_factor,
    target_length=trunk_seqlen
)

# Hold-out test set for final evaluation
dataset_test = BigWigTracksOnlyDataset(
    bigwig_folder='./bws/',
    enformer_loci_path=f'cv_splits_{bw_seq_len}/fold_1_test_loci.bed',
    fasta_file=config['genome_fasta'],
    annot_file='bigwig_track_means.csv',
    ref='ASM584v2',
    downsample_factor=down_sample_factor,
    target_length=trunk_seqlen
)

# Data augmentation functions
def reverse_complement_sequence(seq):
    """Apply reverse complement to one-hot encoded DNA sequence."""
    # seq shape: (batch, channels, length) where channels = [A, C, G, T]
    # Reverse complement: A<->T, C<->G, and reverse the sequence
    complement_map = torch.tensor([3, 2, 1, 0])  # A->T, C->G, G->C, T->A
    seq_rc = seq[:, complement_map, :].flip(-1)  # Complement and reverse
    return seq_rc

def augment_batch(sequences, targets, p=0.5):
    """Apply reverse complement augmentation to batch."""
    if not CONFIG['augment_data']:
        return sequences, targets
    
    batch_size = sequences.size(0)
    mask = torch.rand(batch_size) < p
    
    if mask.any():
        sequences_aug = sequences.clone()
        sequences_aug[mask] = reverse_complement_sequence(sequences[mask])
        return sequences_aug, targets
    
    return sequences, targets

# Metrics calculation
def compute_metrics(predictions, targets):
    """Compute genomics-relevant metrics."""
    predictions_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # Overall metrics
    try:
        overall_corr, _ = pearsonr(predictions_np.flatten(), targets_np.flatten())
        metrics['pearson_r'] = overall_corr if not np.isnan(overall_corr) else 0.0
    except:
        metrics['pearson_r'] = 0.0
    
    try:
        metrics['r2_score'] = r2_score(targets_np.flatten(), predictions_np.flatten())
    except:
        metrics['r2_score'] = 0.0
    
    # Per-target metrics (important for multi-task)
    per_target_r = []
    for i in range(targets.size(1)):
        try:
            corr, _ = pearsonr(predictions_np[:, i], targets_np[:, i])
            per_target_r.append(corr if not np.isnan(corr) else 0.0)
        except:
            per_target_r.append(0.0)
    
    metrics['mean_target_r'] = np.mean(per_target_r)
    metrics['per_target_r'] = per_target_r
    
    return metrics

# Create data loaders with proper batch sizes
train_dataloader = get_bigwig_tracks_dataloader(
    dataset_train, batch_size=CONFIG['batch_size'], cycle_iter=False
)
val_dataloader = get_bigwig_tracks_dataloader(
    dataset_val, batch_size=CONFIG['batch_size'], cycle_iter=False
)
test_dataloader = get_bigwig_tracks_dataloader(
    dataset_test, batch_size=CONFIG['batch_size'], cycle_iter=False
)


# Model creation
if MODEL == 'ALPHAGENOME':
    model = create_mini_alphagenome(base_dim=32, layers=nlayers, num_targets=dataset_train.ntargets)
elif MODEL == 'BOLTZNET':
    assert bw_seq_len == 128
    model = BoltzNet(
        ntargets=dataset_train.ntargets, 
        sequence_length=bw_seq_len,
        l1_reg=1e-5  # Add L1 regularization
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = model.to(device)

# Optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    betas=(0.9, 0.999),
    eps=1e-8
)

# Learning rate scheduler with warmup
warmup_scheduler = LinearLR(
    optimizer, 
    start_factor=0.1, 
    end_factor=1.0, 
    total_iters=CONFIG['warmup_epochs']
)
cosine_scheduler = CosineAnnealingLR(
    optimizer, 
    T_max=CONFIG['num_epochs'] - CONFIG['warmup_epochs'],
    eta_min=CONFIG['learning_rate'] * 0.01
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[CONFIG['warmup_epochs']]
)

# Mixed precision scaler
scaler = GradScaler() if CONFIG['use_mixed_precision'] else None

# Early stopping
best_val_loss = float('inf')
patience_counter = 0

# Checkpoint directory
if CONFIG['save_checkpoints']:
    Path(CONFIG['checkpoint_dir']).mkdir(exist_ok=True)

# Initialize wandb with proper config
if USE_WANDB:
    wandb.init(
        project="tf-bind-transformer",
        name=f"genomic-{MODEL.lower()}-{CONFIG['batch_size']}bs",
        config={
            **CONFIG,  # Include all config parameters
            "model_type": MODEL,
            "num_targets": dataset_train.ntargets,
            "device": str(device),
            "sequence_length": bw_seq_len,
            "downsample_factor": down_sample_factor,
            "trunk_seqlen": trunk_seqlen,
        }
    )

track_means = torch.ones(dataset_train.ntargets) * 2.0

print(f"Number of targets: {dataset_train.ntargets}")
model_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {model_params:,}")

# Log model parameters to wandb
if USE_WANDB:
    wandb.log({"model_parameters": model_params})

criterion = torch.nn.MSELoss()
step = 0

print("Starting training...")
print(f"Training for {CONFIG['num_epochs']} epochs with early stopping patience {CONFIG['patience']}")

for epoch in range(CONFIG['num_epochs']):
    # Training phase
    model.train()
    train_loss = 0.0
    train_metrics_sum = {'pearson_r': 0.0, 'r2_score': 0.0, 'mean_target_r': 0.0}
    num_batches = 0
    
    for batch_idx, (sequences, targets) in enumerate(train_dataloader):
        # Data augmentation
        sequences, targets = augment_batch(sequences, targets)
        
        # Move to device
        sequences = sequences.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if CONFIG['use_mixed_precision']:
            with autocast():
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                
                # Add L1 regularization for BoltzNet
                if MODEL == 'BOLTZNET':
                    loss += model.l1_loss()
        else:
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            # Add L1 regularization for BoltzNet
            if MODEL == 'BOLTZNET':
                loss += model.l1_loss()
        
        # Backward pass with gradient clipping
        if CONFIG['use_mixed_precision']:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
            optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        num_batches += 1
        step += 1
        
        # Compute detailed metrics every few batches
        if batch_idx % 10 == 0:
            with torch.no_grad():
                batch_metrics = compute_metrics(predictions, targets)
                for key in train_metrics_sum:
                    if key in batch_metrics:
                        train_metrics_sum[key] += batch_metrics[key]
        
        # Debug info for first batch
        if batch_idx == 0 and epoch == 0:
            print(f"Input shapes - Sequences: {sequences.shape}, Targets: {targets.shape}")
            print(f"Output shape - Predictions: {predictions.shape}")
            print(f"Device: {sequences.device}")
    
    # Calculate epoch metrics
    avg_train_loss = train_loss / num_batches if num_batches > 0 else 0.0
    avg_train_metrics = {k: v / max(1, num_batches // 10) for k, v in train_metrics_sum.items()}
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_predictions = []
    val_targets = []
    val_num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(val_dataloader):
            sequences = sequences.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            if CONFIG['use_mixed_precision']:
                with autocast():
                    predictions = model(sequences)
                    loss = criterion(predictions, targets)
            else:
                predictions = model(sequences)
                loss = criterion(predictions, targets)
            
            val_loss += loss.item()
            val_num_batches += 1
            
            # Collect predictions for metrics
            val_predictions.append(predictions.cpu())
            val_targets.append(targets.cpu())
    
    # Calculate validation metrics
    avg_val_loss = val_loss / val_num_batches if val_num_batches > 0 else float('inf')
    
    if val_predictions:
        val_predictions_all = torch.cat(val_predictions, dim=0)
        val_targets_all = torch.cat(val_targets, dim=0)
        val_metrics = compute_metrics(val_predictions_all, val_targets_all)
    else:
        val_metrics = {'pearson_r': 0.0, 'r2_score': 0.0, 'mean_target_r': 0.0}
    
    # Learning rate step
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Print epoch results
    print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}:")
    print(f"  Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    print(f"  Train R: {avg_train_metrics['pearson_r']:.3f}, Val R: {val_metrics['pearson_r']:.3f}")
    print(f"  Val R²: {val_metrics['r2_score']:.3f}, Mean Target R: {val_metrics['mean_target_r']:.3f}")
    print(f"  LR: {current_lr:.2e}")
    
    # Wandb logging
    if USE_WANDB:
        log_dict = {
            "epoch": epoch + 1,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss,
            "val/pearson_r": val_metrics['pearson_r'],
            "val/r2_score": val_metrics['r2_score'],
            "val/mean_target_r": val_metrics['mean_target_r'],
            "lr": current_lr,
        }
        # Add training metrics
        for key, value in avg_train_metrics.items():
            log_dict[f"train/{key}"] = value
        
        wandb.log(log_dict)
    
    # Early stopping and checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        
        # Save best model
        if CONFIG['save_checkpoints']:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'val_metrics': val_metrics,
                'config': CONFIG,
            }
            torch.save(checkpoint, f"{CONFIG['checkpoint_dir']}/best_model.pt")
            print(f"  ✓ New best model saved (Val Loss: {avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs (patience: {CONFIG['patience']})")
            break

# Final test evaluation with best model
if CONFIG['save_checkpoints'] and Path(f"{CONFIG['checkpoint_dir']}/best_model.pt").exists():
    print("\nLoading best model for final test evaluation...")
    checkpoint = torch.load(f"{CONFIG['checkpoint_dir']}/best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']} (Val Loss: {checkpoint['val_loss']:.4f})")

# Test evaluation
model.eval()
test_predictions = []
test_targets = []

print("Running final test evaluation...")
with torch.no_grad():
    for sequences, targets in test_dataloader:
        sequences = sequences.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        if CONFIG['use_mixed_precision']:
            with autocast():
                predictions = model(sequences)
        else:
            predictions = model(sequences)
        
        test_predictions.append(predictions.cpu())
        test_targets.append(targets.cpu())

if test_predictions:
    test_predictions_all = torch.cat(test_predictions, dim=0)
    test_targets_all = torch.cat(test_targets, dim=0)
    test_metrics = compute_metrics(test_predictions_all, test_targets_all)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS")
    print("="*50)
    print(f"Test Pearson R: {test_metrics['pearson_r']:.4f}")
    print(f"Test R² Score: {test_metrics['r2_score']:.4f}")
    print(f"Mean Target R: {test_metrics['mean_target_r']:.4f}")
    print(f"Per-target correlations: {[f'{r:.3f}' for r in test_metrics['per_target_r'][:5]]}")
    
    if USE_WANDB:
        wandb.log({
            "final_test/pearson_r": test_metrics['pearson_r'],
            "final_test/r2_score": test_metrics['r2_score'],
            "final_test/mean_target_r": test_metrics['mean_target_r'],
        })

print("\nTraining completed!")
if USE_WANDB:
    wandb.finish()

