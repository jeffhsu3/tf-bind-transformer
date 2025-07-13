# Slight refactoring of dataset_bigwig to be more focused on kfold CVs rather than
# chromosome test train splits
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
import torch
from torch.utils.data import DataLoader, Dataset


def exists(val):
    return val is not None


from enformer_pytorch import FastaInterval

from tf_bind_transformer.data import (
    cycle,
    filter_by_col_isin,
    pl_isin,
    pl_notin,
    read_bed,
)


class BigWigTracksOnlyDataset(Dataset):
    """
    Dataset for loading and processing BigWig data for tracks only, without
    transcription factor protein sequences.
    """

    def __init__(
        self,
        *,
        bigwig_folder,
        enformer_loci_path,
        fasta_file,
        ref,
        annot_file=None,
        filter_chromosome_ids=None,
        downsample_factor=128,
        target_length=896,
        bigwig_reduction_type="sum",
        filter_sequences_by=None,
        **kwargs,
    ):
        super().__init__()

        if not exists(bigwig_folder):
            self.invalid = True
            self.ntargets = 0
            return

        bigwig_folder = Path(bigwig_folder)
        assert bigwig_folder.exists(), "bigwig folder does not exist"

        bw_experiments = [p.stem for p in bigwig_folder.glob("*.bw")]
        print(bw_experiments)
        assert len(bw_experiments) > 0, "no bigwig files found in bigwig folder"
        loci = read_bed(enformer_loci_path)

        if annot_file is not None:
            annot_df = pl.read_csv(
                annot_file,
                separator="\t",
                has_header=False,
                columns=list(map(lambda i: f"column_{i + 1}", range(4))),
            )
            #annot_df = annot_df.filter(pl.col("column_2") == ref)
            annot_df = filter_by_col_isin(annot_df, "column_2", bw_experiments)
            # Reorder df to match bw_experiments. AI!
            self.annot = annot_df

        print(annot_df)

        if exists(filter_sequences_by):
            col_name, col_val = filter_sequences_by
            loci = loci.filter(pl.col(col_name) == col_val)

        self.fasta = FastaInterval(fasta_file=fasta_file, **kwargs)
        self.df = loci
        self.ntargets = len(bw_experiments)

        # bigwigs
        self.bigwigs = [
            (str(i), pyBigWig.open(str(bigwig_folder / f"{str(i)}.bw")))
            for i in bw_experiments
        ]

        self.downsample_factor = downsample_factor
        self.target_length = target_length

        self.bigwig_reduction_type = bigwig_reduction_type
        self.invalid = False

    def __len__(self):
        if self.invalid:
            return 0

        return len(self.df) * int(self.ntargets > 0)

    def __getitem__(self, ind):
        chr_name, begin, end, _ = self.df.row(ind)
        seq = self.fasta(chr_name, begin, end)

        # calculate bigwig
        # properly downsample and then crop
        all_bw_values = []

        for bw_path, bw in self.bigwigs:
            try:
                bw_values = bw.values(chr_name, begin, end)
                all_bw_values.append(bw_values)
            except:
                print(
                    f"hitting invalid range for {bw_path} - ({chr_name}, {begin}, {end})"
                )
                exit()

        output = np.stack(all_bw_values, axis=-1)
        output = output.reshape((-1, self.downsample_factor, self.ntargets))

        if self.bigwig_reduction_type == "mean":
            om = np.nanmean(output, axis=1)
        elif self.bigwig_reduction_type == "sum":
            om = np.nansum(output, axis=1)
        else:
            raise ValueError(f"unknown reduction type {self.bigwig_reduction_type}")

        output_length = om.shape[0]

        if output_length < self.target_length:
            raise ValueError(f"target length {self.target_length} cannot be less than the output length {output_length}")

        if output_length > self.target_length:
            trim = (output_length - self.target_length) // 2
            om = om[trim:trim + self.target_length]
        # If output_length == target_length, no trimming needed

        np.nan_to_num(om, copy=False)

        label = torch.Tensor(om)
        return seq, label


def get_bigwig_tracks_dataloader(ds, cycle_iter=False, **kwargs):
    """
    Returns a DataLoader for BigWigTracksOnlyDataset.

    Args:
        ds (BigWigTracksOnlyDataset): The dataset.
        cycle_iter (bool, optional): Whether to cycle the iterator. Defaults to False.
        **kwargs: Additional arguments for DataLoader.

    Returns:
        torch.utils.data.DataLoader: The DataLoader instance.
    """
    dataset_len = len(ds)
    batch_size = kwargs.get("batch_size")
    drop_last = dataset_len > batch_size

    dl = DataLoader(ds, drop_last=drop_last, **kwargs)
    wrapper = cycle if cycle_iter else iter
    return wrapper(dl)
