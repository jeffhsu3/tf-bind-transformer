#!/usr/bin/env python3
"""
Script to merge paired BigWig files (_1 and _2 replicates) and reheader chromosome names.

This script:
1. Reads all BigWig files in a specified directory
2. Identifies files that can be paired (_1 and _2 replicates)
3. Merges paired files by averaging their values
4. Calculates Spearman correlation between paired files
5. Reheaders merged files to use 'U00096.3' instead of 'NC_000913.3'
6. Saves a summary dataframe with original files, output files, and correlations
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import pyBigWig
from pathlib import Path
from scipy.stats import spearmanr
import collections
import tempfile
import subprocess
from typing import Dict, List, Tuple, Optional
import argparse

SPECIFIC_MAPPING = {
    'ERR475431': 'FliA_2'
}

def find_bigwig_files(bigwig_folder: str) -> List[str]:
    """Find all BigWig files in the specified folder."""
    bw_pattern = os.path.join(bigwig_folder, "*.bw")
    bigwig_pattern = os.path.join(bigwig_folder, "*.bigwig")
    
    files = glob.glob(bw_pattern) + glob.glob(bigwig_pattern)
    return sorted(files)

def parse_filename(filename: str) -> Tuple[str, Optional[str]]:
    """
    Parse filename to extract TF name and replicate number.
    
    Returns:
        Tuple of (tf_name, replicate) where replicate is '1', '2', or None
    """
    basename = os.path.basename(filename)
    # Remove extension
    name = basename.replace('.bw', '').replace('.bigwig', '')
    
    # Check for _1 or _2 suffix
    if name.endswith('_1'):
        return name[:-2], '1'
    elif name.endswith('_2'):
        return name[:-2], '2'
    else:
        return name, None

def group_files_by_tf(files: List[str]) -> Dict[str, Dict[str, str]]:
    """
    Group files by transcription factor name.
    
    Returns:
        Dict mapping TF name to dict of {'1': filepath, '2': filepath}
    """
    tf_groups = {}
    
    for file_path in files:
        tf_name, replicate = parse_filename(file_path)
        
        if tf_name not in tf_groups:
            tf_groups[tf_name] = {}
            
        if replicate:
            tf_groups[tf_name][replicate] = file_path
    
    return tf_groups

def merge_bigwig_files(bw1_path: str, bw2_path: str, output_path: str, step_size: int) -> Tuple[bool, float]:
    """
    Merge two BigWig files by averaging their values over given intervals.
    
    Args:
        bw1_path: Path to first BigWig file
        bw2_path: Path to second BigWig file  
        output_path: Path for merged output file
        step_size: Interval size for averaging (bin size).
        
    Returns:
        Tuple of (success, spearman_correlation)
    """
    try:
        with pyBigWig.open(bw1_path) as bw1, \
             pyBigWig.open(bw2_path) as bw2, \
             pyBigWig.open(output_path, "w") as bw_out:

            chroms1 = bw1.chroms()
            chroms2 = bw2.chroms()
            
            if not chroms1 or not chroms2:
                print(f"Warning: Empty chromosome info in {bw1_path} or {bw2_path}")
                return False, np.nan
                
            common_chroms = {k: v for k, v in chroms1.items() if k in chroms2 and chroms1[k] == chroms2[k]}
            if not common_chroms:
                print(f"Warning: No common chromosomes found between {bw1_path} and {bw2_path}")
                return False, np.nan

            header = []
            output_chrom_map = {}
            for chrom, length in common_chroms.items():
                output_chrom = 'U00096.3' if chrom == 'NC_000913.3' else chrom
                output_chrom_map[chrom] = output_chrom
                header.append((output_chrom, length))
            bw_out.addHeader(header)

            all_means1 = []
            all_means2 = []

            for chrom, length in common_chroms.items():
                output_chrom = output_chrom_map[chrom]
                
                starts, ends, values = [], [], []

                for start in range(0, length, step_size):
                    end = min(start + step_size, length)
                    if start >= end:
                        continue

                    # Get mean values for this interval, ignoring NaNs
                    mean1 = bw1.stats(chrom, start, end, type="mean")[0]
                    mean2 = bw2.stats(chrom, start, end, type="mean")[0]

                    avg_val = None
                    if mean1 is not None and mean2 is not None:
                        avg_val = (mean1 + mean2) / 2
                        all_means1.append(mean1)
                        all_means2.append(mean2)
                    elif mean1 is not None:
                        avg_val = mean1
                    elif mean2 is not None:
                        avg_val = mean2
                    
                    if avg_val is not None and avg_val > 0:
                        starts.append(start)
                        ends.append(end)
                        values.append(avg_val)

                if starts:
                    bw_out.addEntries([output_chrom] * len(starts), starts, ends=ends, values=values)

            correlation = np.nan
            if len(all_means1) >= 2:
                # spearmanr returns nan for constant input
                with np.errstate(invalid='ignore'):
                    correlation, _ = spearmanr(all_means1, all_means2)
                    if np.isnan(correlation):
                        correlation = 0.0 # if std dev is 0, consider correlation 0.

            return True, correlation

    except Exception as e:
        print(f"Error merging {bw1_path} and {bw2_path}: {e}")
        return False, np.nan

def capitalize_tf_name(tf_name: str) -> str:
    """
    Capitalize first and last letters of a TF name.
    If the name contains an underscore, only the part before the first underscore is modified.
    
    Args:
        tf_name: Original TF name
        
    Returns:
        Capitalized TF name. e.g. 'gade' -> 'GadE', 'some_tf' -> 'SomE_tf'
    """
    parts = tf_name.split('_', 1)
    name_to_capitalize = parts[0]

    if len(name_to_capitalize) >= 2:
        capitalized_part = name_to_capitalize[0].upper() + name_to_capitalize[1:-1] + name_to_capitalize[-1].upper()
    elif len(name_to_capitalize) == 1:
        capitalized_part = name_to_capitalize.upper()
    else:
        capitalized_part = name_to_capitalize
    
    if len(parts) > 1:
        return capitalized_part + '_' + parts[1]
    else:
        return capitalized_part

def main():
    parser = argparse.ArgumentParser(description='Merge and reheader ChIP-seq BigWig files')
    parser.add_argument('--bigwig_folder', 
                       default='/home/jeff/iv3/repos/ChIPdb/data/e_coli/NC_000913.3/bw/',
                       help='Path to folder containing BigWig files')
    parser.add_argument('--output_csv', 
                       default='merged_bigwig_summary.csv',
                       help='Output CSV file with merge summary')
    
    args = parser.parse_args()
    
    bigwig_folder = args.bigwig_folder
    
    print(f"Processing BigWig files in: {bigwig_folder}")
    
    # Find all BigWig files
    files = find_bigwig_files(bigwig_folder)
    print(f"Found {len(files)} BigWig files")
    
    if not files:
        print("No BigWig files found!")
        return
    
    # Group files by TF
    tf_groups = group_files_by_tf(files)
    print(f"Found {len(tf_groups)} unique TF groups")
    
    # Create results dataframe
    results = []
    
    # Process each TF group
    for tf_name, replicate_files in tf_groups.items():
        print(f"\nProcessing TF: {tf_name}")
        
        # Check if we have both replicates
        if '1' in replicate_files and '2' in replicate_files:
            file1 = replicate_files['1']
            file2 = replicate_files['2']
            
            print(f"  Found paired files:")
            print(f"    Replicate 1: {os.path.basename(file1)}")
            print(f"    Replicate 2: {os.path.basename(file2)}")
            
            # Create output filename
            capitalized_tf = capitalize_tf_name(tf_name)
            output_filename = f"{capitalized_tf}_merged.bw"
            output_path = os.path.join(bigwig_folder, output_filename)
            output_path = os.path.join('/home/jeff/iv3/repos/tf-bind-transformer/tests/ChIPdb', output_filename)
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"    Skipping {output_filename}, already exists.")
                results.append({
                    'original_file_1': os.path.basename(file1),
                    'original_file_2': os.path.basename(file2),
                    'output_merged_file': output_filename,
                    'spearman_correlation': np.nan,
                    'tf_name': tf_name,
                    'capitalized_tf_name': capitalized_tf
                })
                continue
            
            # Merge files
            success, correlation = merge_bigwig_files(
                file1,
                file2,
                output_path,
                step_size=1,
            )

            if not np.isnan(correlation):
                print(f"    Spearman correlation: {correlation:.4f}")
            
            if success:
                print(f"    Successfully merged to: {output_filename}")
                results.append({
                    'original_file_1': os.path.basename(file1),
                    'original_file_2': os.path.basename(file2),
                    'output_merged_file': output_filename,
                    'spearman_correlation': correlation,
                    'tf_name': tf_name,
                    'capitalized_tf_name': capitalized_tf
                })
            else:
                print(f"    Failed to merge files")
                results.append({
                    'original_file_1': os.path.basename(file1),
                    'original_file_2': os.path.basename(file2),
                    'output_merged_file': 'FAILED',
                    'spearman_correlation': correlation,
                    'tf_name': tf_name,
                    'capitalized_tf_name': capitalized_tf
                })
        else:
            print(f"  Skipping {tf_name}: Missing replicate(s)")
            # Record unpaired files
            for rep, filepath in replicate_files.items():
                results.append({
                    'original_file_1': os.path.basename(filepath) if rep == '1' else '',
                    'original_file_2': os.path.basename(filepath) if rep == '2' else '',
                    'output_merged_file': 'UNPAIRED',
                    'spearman_correlation': np.nan,
                    'tf_name': tf_name,
                    'capitalized_tf_name': capitalize_tf_name(tf_name)
                })
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    
    print(f"\n=== SUMMARY ===")
    print(f"Total TF groups processed: {len(tf_groups)}")
    print(f"Successfully merged: {len(df[df['output_merged_file'].str.endswith('.bw')])}")
    print(f"Failed merges: {len(df[df['output_merged_file'] == 'FAILED'])}")
    print(f"Unpaired files: {len(df[df['output_merged_file'] == 'UNPAIRED'])}")
    print(f"Results saved to: {args.output_csv}")
    
    # Show correlation statistics
    correlations = df['spearman_correlation'].dropna()
    if len(correlations) > 0:
        print(f"\nCorrelation statistics:")
        print(f"  Mean: {correlations.mean():.4f}")
        print(f"  Median: {correlations.median():.4f}")
        print(f"  Min: {correlations.min():.4f}")
        print(f"  Max: {correlations.max():.4f}")

if __name__ == "__main__":
    main()
