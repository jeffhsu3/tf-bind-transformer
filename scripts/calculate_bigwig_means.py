#!/usr/bin/env python3
"""
Script to calculate track means from all bigwig files in a folder.
Reads all .bw and .bigwig files from ./bws/ and saves track means to a pickle file.
"""

import os
import glob
import pyBigWig
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_bigwig_means(folder_path="./bws/", output_file="bigwig_track_means.pkl"):
    """
    Calculate track means for all bigwig files in the specified folder.
    
    Args:
        folder_path (str): Path to folder containing bigwig files
        output_file (str): Output pickle filename
    
    Returns:
        pd.DataFrame: DataFrame with track means
    """
    
    # Find all bigwig files
    bigwig_patterns = ["*.bw", "*.bigwig", "*.bigWig"]
    bigwig_files = []
    
    for pattern in bigwig_patterns:
        bigwig_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    
    if not bigwig_files:
        logger.warning(f"No bigwig files found in {folder_path}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(bigwig_files)} bigwig files")
    
    results = []
    
    for file_path in bigwig_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing: {filename}")
        
        try:
            # Open bigwig file
            bw = pyBigWig.open(file_path)
            
            if bw.isBigWig():
                # Get chromosome information
                chromosomes = bw.chroms()
                
                # Calculate overall mean across all chromosomes
                total_sum = 0.0
                total_length = 0
                
                for chrom, length in chromosomes.items():
                    try:
                        # Get values for entire chromosome
                        values = bw.values(chrom, 0, length)
                        
                        # Filter out NaN values
                        values = np.array([v for v in values if not np.isnan(v)])
                        
                        if len(values) > 0:
                            total_sum += np.sum(values)
                            total_length += len(values)
                    
                    except Exception as e:
                        logger.warning(f"Error processing {chrom} in {filename}: {e}")
                        continue
                
                if total_length > 0:
                    mean_value = total_sum / total_length
                else:
                    mean_value = np.nan
                    logger.warning(f"No valid data found in {filename}")
                
                # Get track name (use filename without extension as track name)
                track_name = os.path.splitext(filename)[0]
                
                results.append({
                    'filename': filename,
                    'track_name': track_name,
                    'mean_value': mean_value,
                    'total_bases': total_length
                })
                
                bw.close()
                
            else:
                logger.warning(f"{filename} is not a valid bigwig file")
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to pickle
    df.to_pickle(output_file)
    logger.info(f"Results saved to {output_file}")
    
    # Print summary
    if not df.empty:
        logger.info(f"Processed {len(df)} tracks")
        logger.info(f"Mean values range: {df['mean_value'].min():.4f} to {df['mean_value'].max():.4f}")
    
    return df

def main():
    """Main function to run the script."""
    bw_folder = 'tests/bws/'
    
    # Check if bws folder exists
    if not os.path.exists(bw_folder):
        logger.error(f"Folder {bw_folder} does not exist")
        return
    
    # Calculate means
    df = calculate_bigwig_means(bw_folder, "bigwig_track_means.pkl")
    
    # Display results
    if not df.empty:
        print("\nTrack Means Summary:")
        print(df[['track_name', 'mean_value', 'total_bases']].to_string(index=False))
        
        # Save additional CSV for easy viewing
        csv_file = "bigwig_track_means.csv"
        df.to_csv(csv_file, index=False, sep="\t")
        logger.info(f"Also saved results to {csv_file} for easy viewing")

if __name__ == "__main__":
    main()
