import unittest
import torch
import pyBigWig
import numpy as np
import polars as pl
from pathlib import Path
import yaml
import rich

from tf_bind_transformer.data_bigwig2 import BigWigTracksOnlyDataset, get_bigwig_tracks_dataloader
from tf_bind_transformer.data import FactorProteinDataset # Assuming this is needed for BigWigDataset
from tf_bind_transformer.mini_alphagenome import create_mini_alphagenome, multinomial_loss

with open('/home/jeff/iv4/ref/ecoli/ecoli.yaml') as f:
    config = yaml.safe_load(f)

dataset = BigWigTracksOnlyDataset(
    #bigwig_folder='tfactor_faa/bigwig/',
    bigwig_folder='./bws/',
    enformer_loci_path='cv_splits/fold_1_train_loci.bed',
    fasta_file=config['genome_fasta'],
    ref='ASM584v2',
)

model = SequenceEncoder(16)
BATCH_SIZE=2
seq_len = 4096

dataloader = get_bigwig_tracks_dataloader(dataset, batch_size=2)
print(dataset.ntargets)
for i, j in dataloader:
    output, intermediates = model(i)

    for key, tensor in intermediates.items():
        print(f"  {key}: {tensor.shape}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    break
