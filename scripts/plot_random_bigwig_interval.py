#!/usr/bin/env python3
"""
This script plots a random 200 bp interval from a given BigWig file using Plotly.
"""
import argparse
import os
import random
import pyBigWig
import plotly.graph_objects as go
import numpy as np

def plot_random_interval(bw_path: str):
    """
    Selects a random 200bp interval from a BigWig file and plots the signal.

    Args:
        bw_path (str): Path to the BigWig file.
    """
    if not os.path.exists(bw_path):
        print(f"Error: BigWig file not found at {bw_path}")
        return

    try:
        bw = pyBigWig.open(bw_path)
    except Exception as e:
        print(f"Error opening BigWig file: {e}")
        return

    chroms = bw.chroms()
    if not chroms:
        print("Error: BigWig file has no chromosomes.")
        bw.close()
        return
        
    # Find a chromosome long enough for a 200bp interval
    valid_chroms = [c for c, length in chroms.items() if length >= 200]
    if not valid_chroms:
        print("Error: No chromosome is long enough for a 200bp interval.")
        bw.close()
        return

    chrom = random.choice(valid_chroms)
    length = chroms[chrom]
    
    start = random.randint(0, length - 200)
    end = start + 200

    try:
        values = bw.values(chrom, start, end)
        # pyBigWig returns nan for missing values, replace with 0 for a continuous plot
        values = np.nan_to_num(values, nan=0.0)
    except RuntimeError as e:
        print(f"Error reading values from BigWig file: {e}")
        bw.close()
        return

    bw.close()

    x_coords = list(range(start, end))

    fig = go.Figure(data=go.Scatter(x=x_coords, y=values, mode='lines', name='Signal'))

    fig.update_layout(
        title=f"BigWig Signal for {os.path.basename(bw_path)}<br>Interval: {chrom}:{start}-{end}",
        xaxis_title=f"Genomic Position on {chrom}",
        yaxis_title="Signal Value",
        template="plotly_white"
    )

    fig.show()

def main():
    parser = argparse.ArgumentParser(description="Plot a random 200bp interval from a BigWig file.")
    parser.add_argument("bigwig_file", help="Path to the BigWig file.")
    args = parser.parse_args()

    plot_random_interval(args.bigwig_file)

if __name__ == "__main__":
    main()
