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

# AI: Fix the following implementation of reading in a bigwig and averaging the values.  Make it at the base pair level if possible.  Also calculate the correlation of the values to be averaged. AI!
def merge_bigwig_files(bw1_path: str, bw2_path: str, output_path: str, step_size: int) -> tuple(bool, float):
    """
    Merge two BigWig files by averaging their values.
    
    Args:
        bw1_path: Path to first BigWig file
        bw2_path: Path to second BigWig file  
        output_path: Path for merged output file
        step_size: Step_size to merge ChiPseq.  Alpha genome merges at 128 interval bins
        
    Returns:
        True if successful, False otherwise
    """
    bw1 = pyBigWig.open(bw1_path)
    bw2 = pyBigWig.open(bw2_path)
    bw_out = pyBigWig.open(output_path, "w")
    
    try:
        # Get chromosome info
        chroms1 = bw1.chroms()
        chroms2 = bw2.chroms()
        
        if not chroms1 or not chroms2:
            print(f"Warning: Empty chromosome info in {bw1_path} or {bw2_path}")
            return False
            
        # Use first chromosome and ensure it exists in both files
        chrom = list(chroms1.keys())[0]
        if chrom not in chroms2:
            print(f"Warning: Chromosome {chrom} not found in both files")
            return False

        assert chroms1[chrom] == chroms2[chrom]
        length = chroms1[chrom]
        correlations = []
        
        for start in range(0, length, step_size):
            end = min(start + step_size, length)
            
            # Get values for this interval
            vals1 = bw1.values(chrom, start, end)
            vals2 = bw2.values(chrom, start, end)
            correlation = spearmanr(vals1, vals2)[0]

            correlations.append(correlation)
            
            # Calculate average, handling None values
            valid_vals = []
            for v1, v2 in zip(vals1, vals2):
                if v1 is not None and v2 is not None and not np.isnan(v1) and not np.isnan(v2):
                    valid_vals.append((v1 + v2) / 2)
                elif v1 is not None and not np.isnan(v1):
                    valid_vals.append(v1)
                elif v2 is not None and not np.isnan(v2):
                    valid_vals.append(v2)
            
            '''
            if valid_vals:
                avg_val = np.mean(valid_vals)
                if avg_val > 0:  # Only write non-zero values
                    tmp_bg.write(f"U00096.3\t{start}\t{end}\t{avg_val:.6f}\n")
            '''
            bw_out.write()

        
        # Convert bedGraph to BigWig with correct chromosome name
        # Create chromosome sizes file
        #tmp_sizes_path = tmp_sizes.name
        #tmp_sizes.write(f"U00096.3\t{length}\n")
        out_corr = np.mean(correlations)
        success = True
        bw1.close()
        bs2.close()
        bw_out.close()
        return success, out_corr 
        
    except Exception as e:
        print(f"Error merging {bw1_path} and {bw2_path}: {e}")
        return False
    finally:
        bw1.close()
        bw2.close()

def capitalize_tf_name(tf_name: str) -> str:
    """
    Capitalize first and second letters of TF name.
    
    Args:
        tf_name: Original TF name
        
    Returns:
        TF name with first two letters capitalized
    """
    if len(tf_name) >= 2:
        return tf_name[0].upper() + tf_name[1].upper() + tf_name[2:]
    elif len(tf_name) == 1:
        return tf_name.upper()
    else:
        return tf_name

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
            
            # Calculate correlation
            #correlation = calculate_spearman_correlation(file1, file2)
            
            # Create output filename
            capitalized_tf = capitalize_tf_name(tf_name)
            output_filename = f"{capitalized_tf}_condition.bw"
            output_path = os.path.join(bigwig_folder, output_filename)
            
            # Merge files
            success, correlation = merge_bigwig_files(
                file1,
                file2,
                output_path,
                step_size=128,
            )
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
