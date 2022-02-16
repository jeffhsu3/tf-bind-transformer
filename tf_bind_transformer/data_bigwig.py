from Bio import SeqIO
from random import choice
from pathlib import Path
import polars as pl

import numpy as np

import torch
#from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.data import FactorProteinDataset, ContextDataset, cast_list, filter_df_by_tfactor_fastas
from tf_bind_transformer.data import pl_isin, pl_notin, fetch_experiments_index, parse_exp_target_cell, read_bed
from enformer_pytorch import FastaInterval

import pyBigWig

def exists(val):
    return val is not None

def get_chr_names(ids):
    return set(map(lambda t: f'chr{t}', ids))

CHR_IDS = set([*range(1, 23), 'X'])
CHR_NAMES = get_chr_names(CHR_IDS)

def chip_atlas_add_experiment_target_cell(df, col="column_4"):
    df = df.clone()

    targets = df.select(col)
    targets = targets.rename({col: 'target'}).to_series(0)
    df.insert_at_idx(2, targets)

    cell_type = df.select("column_5")
    cell_type = cell_type.rename({"column_5": 'cell_type'}).to_series(0)
    df.insert_at_idx(2, cell_type)
    
    return df


# dataset for CHIP ATLAS - all peaks
class BigWigDataset(Dataset):
    def __init__(
        self,
        *,
        factor_fasta_folder,
        bigwig_folder,
        enformer_loci_path,
        annot_file = None,
        filter_chromosome_ids = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        df_frac = 1.,
        experiments_json_path = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        **kwargs
    ):
        super().__init__()
        assert exists(annot_file) 

        bw_experiments = [str(i).split("/")[-1].rstrip(".bw") for i in Path(bigwig_folder).glob('*.bw')]
        loci = read_bed(enformer_loci_path)

        annot_df = pl.read_csv(annot_file, sep = "\t", has_header = False, columns=range(17))
        only_ref = ["mm10", "hg38"]
        annot_df = annot_df.filter(pl_isin("column_2", only_ref))
        # :TODO find out why this step is taking forever
        annot_df = annot_df.filter(pl_isin("column_1", bw_experiments[0:25]))
        
        if df_frac < 1:
            annot_df = annot_df.sample(frac = df_frac)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        loci = loci.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)
        annot_df = chip_atlas_add_experiment_target_cell(annot_df)
        annot_df = filter_df_by_tfactor_fastas(annot_df, factor_fasta_folder)

        # filter dataset by inclusion and exclusion list of targets
        # (<all available targets> intersect <include targets>) subtract <exclude targets>

        include_targets = cast_list(include_targets)
        exclude_targets = cast_list(exclude_targets)

        if include_targets:
            annot_df = annot_df.filter(pl_isin('target', include_targets))

        if exclude_targets:
            annot_df = annot_df.filter(pl_notin('target', exclude_targets))

        # filter dataset by inclusion and exclusion list of cell types
        # same logic as for targets

        include_cell_types = cast_list(include_cell_types)
        exclude_cell_types = cast_list(exclude_cell_types)

        # :TODO reformulate this
        # Cell_type should probably be column_6
        if include_cell_types:
            annot_df = annot_df.filter(pl_isin('cell_type', include_cell_types))

        if exclude_cell_types:
            annot_df = annot_df.filter(pl_notin('cell_type', exclude_cell_types))

        assert len(annot_df) > 0, 'dataset is empty by filter criteria'

        self.fasta = FastaInterval(**kwargs)
        self.df = loci
        self.annot = annot_df
        self.ntargets = self.annot.shape[0]
        # bigwigs
        self.bigwigs = [pyBigWig.open(f"{bigwig_folder}{str(i)}.bw") for i in self.annot.select("column_1").to_series(0)]
        
        # self.experiments_index = fetch_experiments_index(experiments_json_path)

        # context string creator
        # This is needs to be altered
        '''
        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )
        '''

    def __len__(self):
        return len(self.df * self.ntargets)

    def __getitem__(self, ind):
        # TODO return all targets from an individual enformer loci
        chr_name, begin, end, _ = self.df.row(ind % self.df.shape[0])
        target = self.annot.select('target').to_series(0)
        ix_target = ind // self.df.shape[0]
    
        #experiment, target, cell_type = parse_exp_target_cell(experiment_target_cell_type)
        seq = self.fasta(chr_name, begin, end)
        aa_seq = self.factor_ds[target[ix_target]]
        
        #context_str = self.context_ds[cell_type]
        # Make this flexible
        context_str = ''
        exp_bw = self.bigwigs[ix_target]
        output = np.array(exp_bw.values(chr_name, begin, end))
        output = output.reshape((1024, 128))
        om = np.nanmean(output, axis=1)
        om = om[64:-64]
        np.nan_to_num(om, copy=False)

        #read_value = torch.Tensor([reading])
        label = torch.Tensor(om)

        return seq, aa_seq, context_str, label