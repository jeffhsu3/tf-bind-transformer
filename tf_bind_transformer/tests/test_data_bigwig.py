import unittest
import torch
import pyBigWig
import numpy as np
import polars as pl
from pathlib import Path
import os
import shutil

from tf_bind_transformer.data_bigwig import BigWigDataset, BigWigTracksOnlyDataset, get_bigwig_dataloader, get_bigwig_tracks_dataloader
from tf_bind_transformer.data import FactorProteinDataset # Assuming this is needed for BigWigDataset


# Helper function to create a dummy BigWig file
def create_dummy_bigwig(path, chrom_sizes, intervals):
    bw = pyBigWig.open(str(path), "w")
    bw.addHeader(list(chrom_sizes.items()))
    for chrom, start, end, value in intervals:
        bw.addEntries([chrom], [start], ends=[end], values=[value])
    bw.close()

# Helper function to create a dummy FASTA file
def create_dummy_fasta(path, sequences):
    with open(path, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n{seq}\n")

# Helper function to create a dummy BED file
def create_dummy_bed(path, entries):
    with open(path, "w") as f:
        for entry in entries:
            f.write("\t".join(map(str, entry)) + "\n")

# Helper function to create a dummy annotation file
def create_dummy_annot_file(path, data):
    df = pl.DataFrame(data)
    df.write_csv(path, separator='\t', has_header=False)


class TestBigWigData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_dir = Path("tf_bind_transformer/tests/data_bigwig_test_temp")
        cls.test_data_dir.mkdir(parents=True, exist_ok=True)

        cls.factor_fasta_folder = cls.test_data_dir / "factor_fastas"
        cls.factor_fasta_folder.mkdir(exist_ok=True)
        cls.bigwig_folder = cls.test_data_dir / "bigwigs"
        cls.bigwig_folder.mkdir(exist_ok=True)

        cls.genome_fasta_file = cls.test_data_dir / "genome.fa"
        cls.enformer_loci_path = cls.test_data_dir / "enformer_loci.bed"
        cls.annot_file_path = cls.test_data_dir / "annotations.tsv"

        # Create dummy factor fasta files
        create_dummy_fasta(cls.factor_fasta_folder / "TF1.human.fasta", {"TF1.human": "MAAAA"})
        create_dummy_fasta(cls.factor_fasta_folder / "TF2.mouse.fasta", {"TF2.mouse": "MEEEE"})


        # Create dummy genome fasta
        genome_seqs = {
            "chr1": "N" * 200000, # Enformer loci are ~114688, target length 896 * 128 = 114688
            "chr2": "N" * 200000,
        }
        create_dummy_fasta(cls.genome_fasta_file, genome_seqs)

        # Create dummy enformer loci
        # format: chr, start, end, gene_name (or other identifier)
        enformer_loci_data = [
            ("chr1", 10000, 10000 + 114688, "locus1_train", 0, "+", "train"),
            ("chr1", 50000, 50000 + 114688, "locus2_valid", 0, "+", "valid"),
            ("chr2", 20000, 20000 + 114688, "locus3_test", 0, "+", "test"),
        ]
        # Save as bed with 7 columns to match expected format by `read_bed` and subsequent filtering
        with open(cls.enformer_loci_path, "w") as f:
            for entry in enformer_loci_data:
                 f.write(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}\t{entry[4]}\t{entry[5]}\t{entry[6]}\n")


        # Create dummy annotation file for BigWigDataset
        # format: column_1 (exp_id), column_2 (ref), column_3, column_4 (target), column_5 (cell_type), ... (17 cols total)
        cls.annot_data = {
            'column_1': ["exp1", "exp2"],
            'column_2': ["hg38", "hg38"],
            'column_3': ["-", "-"],
            'column_4': ["TF1", "TF2"], # Targets
            'column_5': ["cellA", "cellB"], # Cell types
            'column_6': ["-", "-"],
            'column_7': ["-", "-"],
            'column_8': ["-", "-"],
            'column_9': ["-", "-"],
            'column_10': ["-", "-"],
            'column_11': ["-", "-"],
            'column_12': ["-", "-"],
            'column_13': ["-", "-"],
            'column_14': ["-", "-"],
            'column_15': ["-", "-"],
            'column_16': ["-", "-"],
            'column_17': ["-", "-"],
        }
        create_dummy_annot_file(cls.annot_file_path, cls.annot_data)

        # Create dummy BigWig files
        chrom_sizes_hg38 = {"chr1": 200000, "chr2": 200000}
        # For exp1 (TF1, cellA)
        # Locus1: chr1:10000-124688. Downsampled target length will be 896. Original length 114688
        intervals_exp1 = [
            ("chr1", 10000, 10000 + 114688, 1.0), # locus1
            ("chr1", 50000, 50000 + 114688, 1.5), # locus2
        ]
        create_dummy_bigwig(cls.bigwig_folder / "exp1.bw", chrom_sizes_hg38, intervals_exp1)

        # For exp2 (TF2, cellB)
        intervals_exp2 = [
            ("chr1", 10000, 10000 + 114688, 2.0), # locus1
            ("chr1", 50000, 50000 + 114688, 2.5), # locus2
        ]
        create_dummy_bigwig(cls.bigwig_folder / "exp2.bw", chrom_sizes_hg38, intervals_exp2)

        # Common dataset parameters
        cls.common_params_dataset = dict(
            factor_fasta_folder=str(cls.factor_fasta_folder),
            bigwig_folder=str(cls.bigwig_folder),
            enformer_loci_path=str(cls.enformer_loci_path),
            fasta_file=str(cls.genome_fasta_file),
            annot_file=str(cls.annot_file_path),
            only_ref=['hg38'],
            target_length = 896, # 114688 / 128
            return_seq_indices = False # from FastaInterval kwargs
        )
        cls.common_params_tracks_dataset = dict(
            bigwig_folder=str(cls.bigwig_folder),
            enformer_loci_path=str(cls.enformer_loci_path),
            fasta_file=str(cls.genome_fasta_file),
            annot_file=str(cls.annot_file_path), # Will be filtered by ref
            ref='hg38',
            target_length = 896,
            return_seq_indices = False # from FastaInterval kwargs
        )


    @classmethod
    def tearDownClass(cls):
        # Clean up mock data
        shutil.rmtree(cls.test_data_dir)
        pass

    def test_bigwig_dataset_initialization(self):
        dataset = BigWigDataset(**self.common_params_dataset)
        self.assertFalse(dataset.invalid)
        self.assertEqual(dataset.ntargets, 2) # exp1 and exp2 matching TF1, TF2

    def test_bigwig_dataset_initialization_no_bigwig_folder(self):
        params = self.common_params_dataset.copy()
        params["bigwig_folder"] = "non_existent_folder"
        dataset = BigWigDataset(**params)
        self.assertTrue(dataset.invalid)
        self.assertEqual(dataset.ntargets, 0)

    def test_bigwig_dataset_initialization_no_annot_file(self):
        params = self.common_params_dataset.copy()
        params["annot_file"] = "non_existent_annot.tsv"
        with self.assertRaises(pl.exceptions.ComputeError): # Polars raises ComputeError for file not found
            BigWigDataset(**params)

    def test_bigwig_dataset_len(self):
        dataset = BigWigDataset(**self.common_params_dataset, filter_sequences_by=('column_7', 'train')) # 1 locus
        # n_loci * n_targets = 1 * 2 = 2
        self.assertEqual(len(dataset), 2)

        dataset_all_loci = BigWigDataset(**self.common_params_dataset) # 3 loci from enformer_loci.bed
        # n_loci * n_targets = 3 * 2 = 6
        self.assertEqual(len(dataset_all_loci), 6)


    def test_bigwig_dataset_getitem(self):
        dataset = BigWigDataset(**self.common_params_dataset, filter_sequences_by=('column_7', 'train'), downsample_factor=128, target_length=896)
        # Locus1 (chr1:10000-124688), exp1 (TF1, cellA, value 1.0)
        # Locus1 (chr1:10000-124688), exp2 (TF2, cellB, value 2.0)

        # ind = 0 -> locus_idx = 0 % 1 = 0 (locus1), target_idx = 0 // 1 = 0 (exp1)
        seq, aa_seq, context_str, label = dataset[0]

        self.assertIsInstance(seq, torch.Tensor)
        self.assertEqual(seq.shape, (114688 // 128 * 2 + 896, 4)) # default context length for FastaInterval
        self.assertEqual(aa_seq, "MAAAA")
        self.assertEqual(context_str, "cellA")
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.shape, (896,))
        # Values are sum-reduced over downsample_factor=128
        # Original value 1.0. So sum is 1.0 * 128 = 128.
        self.assertTrue(torch.allclose(label, torch.full((896,), 1.0 * 128)))


        # ind = 1 -> locus_idx = 1 % 1 = 0 (locus1), target_idx = 1 // 1 = 1 (exp2)
        seq2, aa_seq2, context_str2, label2 = dataset[1]
        self.assertEqual(aa_seq2, "MEEEE")
        self.assertEqual(context_str2, "cellB")
        self.assertTrue(torch.allclose(label2, torch.full((896,), 2.0 * 128)))


    def test_bigwig_dataset_reduction_mean(self):
        dataset = BigWigDataset(**self.common_params_dataset, filter_sequences_by=('column_7', 'train'),
                                downsample_factor=128, target_length=896, bigwig_reduction_type='mean')
        # ind = 0 -> locus_idx = 0 (locus1), target_idx = 0 (exp1, value 1.0)
        _, _, _, label = dataset[0]
        self.assertTrue(torch.allclose(label, torch.full((896,), 1.0)))


    def test_bigwig_tracks_only_dataset_initialization(self):
        dataset = BigWigTracksOnlyDataset(**self.common_params_tracks_dataset)
        self.assertFalse(dataset.invalid)
        self.assertEqual(dataset.ntargets, 2) # exp1, exp2

    def test_bigwig_tracks_only_dataset_len(self):
        dataset = BigWigTracksOnlyDataset(**self.common_params_tracks_dataset, filter_sequences_by=('column_7', 'train')) # 1 locus
        # n_loci * (1 if ntargets > 0 else 0) = 1 * 1 = 1
        self.assertEqual(len(dataset), 1)

        dataset_all_loci = BigWigTracksOnlyDataset(**self.common_params_tracks_dataset) # 3 loci
        self.assertEqual(len(dataset_all_loci), 3)


    def test_bigwig_tracks_only_dataset_getitem(self):
        dataset = BigWigTracksOnlyDataset(**self.common_params_tracks_dataset, filter_sequences_by=('column_7', 'train'),
                                          downsample_factor=128, target_length=896)
        # Locus1 (chr1:10000-124688). Values from exp1 (1.0) and exp2 (2.0)
        seq, label = dataset[0] # Only one locus due to filter_sequences_by

        self.assertIsInstance(seq, torch.Tensor)
        self.assertEqual(seq.shape, (114688 // 128 * 2 + 896, 4))
        self.assertIsInstance(label, torch.Tensor)
        self.assertEqual(label.shape, (896, 2)) # target_length, ntargets

        # Values are sum-reduced over downsample_factor=128
        expected_label_exp1 = torch.full((896,), 1.0 * 128)
        expected_label_exp2 = torch.full((896,), 2.0 * 128)
        self.assertTrue(torch.allclose(label[:, 0], expected_label_exp1))
        self.assertTrue(torch.allclose(label[:, 1], expected_label_exp2))

    def test_bigwig_dataloader(self):
        dataset = BigWigDataset(**self.common_params_dataset, filter_sequences_by=('column_7', 'train')) # 2 items
        dataloader = get_bigwig_dataloader(dataset, batch_size=2)

        batch_seq, batch_aa_seq, batch_context_str, batch_labels = next(iter(dataloader))

        self.assertIsInstance(batch_seq, torch.Tensor)
        self.assertEqual(batch_seq.shape, (2, 114688 // 128 * 2 + 896, 4))
        self.assertIsInstance(batch_aa_seq, tuple)
        self.assertEqual(len(batch_aa_seq), 2)
        self.assertIsInstance(batch_context_str, tuple)
        self.assertEqual(len(batch_context_str), 2)
        self.assertIsInstance(batch_labels, torch.Tensor)
        self.assertEqual(batch_labels.shape, (2, 896))

    def test_bigwig_tracks_dataloader(self):
        # Use 2 loci for this test to have batch_size of 2
        # Create a temporary loci file with 2 'train' items
        temp_loci_data = [
            ("chr1", 10000, 10000 + 114688, "locus1_train", 0, "+", "train"),
            ("chr1", 50000, 50000 + 114688, "locus2_train", 0, "+", "train"), # Changed to train
        ]
        temp_loci_path = self.test_data_dir / "temp_enformer_loci_train.bed"
        with open(temp_loci_path, "w") as f:
            for entry in temp_loci_data:
                 f.write(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{entry[3]}\t{entry[4]}\t{entry[5]}\t{entry[6]}\n")

        params = self.common_params_tracks_dataset.copy()
        params["enformer_loci_path"] = str(temp_loci_path)

        dataset = BigWigTracksOnlyDataset(**params, filter_sequences_by=('column_7', 'train')) # 2 items now
        self.assertEqual(len(dataset), 2) # Ensure dataset has 2 items

        dataloader = get_bigwig_tracks_dataloader(dataset, batch_size=2)

        batch_seq, batch_labels = next(iter(dataloader))

        self.assertIsInstance(batch_seq, torch.Tensor)
        self.assertEqual(batch_seq.shape, (2, 114688 // 128 * 2 + 896, 4))
        self.assertIsInstance(batch_labels, torch.Tensor)
        self.assertEqual(batch_labels.shape, (2, 896, 2))

        os.remove(temp_loci_path)


if __name__ == "__main__":
    unittest.main()
