import argparse
import polars as pl
from pyfaidx import Fasta
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import random

def generate_random_loci(fasta_file: str, num_loci: int, locus_length: int, seed: int) -> pl.DataFrame:
    """
    Generates random genomic loci from a FASTA file.

    Args:
        fasta_file (str): Path to the FASTA file.
        num_loci (int): Number of random loci to generate.
        locus_length (int): Length of each locus.
        seed (int): Random seed for reproducibility.

    Returns:
        pl.DataFrame: A DataFrame with columns ['chrom', 'start', 'end', 'name'].
    """
    np.random.seed(seed)
    random.seed(seed)

    fasta = Fasta(fasta_file)
    loci_list = []

    chrom_keys = sorted(list(fasta.keys())) # Ensure consistent order

    attempts = 0
    max_attempts_per_locus = 100 # To prevent infinite loops if space is tight

    for i in range(num_loci):
        locus_generated = False
        for _ in range(max_attempts_per_locus):
            chrom_name = random.choice(chrom_keys)
            chromosome = fasta[chrom_name]
            chrom_len = len(chromosome)

            if chrom_len < locus_length:
                continue

            start = np.random.randint(0, chrom_len - locus_length + 1)
            end = start + locus_length
            loci_list.append({
                "chrom": chrom_name,
                "start": start,
                "end": end,
                "name": f"random_locus_{i+1}"
            })
            locus_generated = True
            break
        if not locus_generated:
            print(f"Warning: Could not generate locus {i+1} after {max_attempts_per_locus} attempts. Chromosome lengths might be too short or num_loci too high.")

    if not loci_list:
        raise ValueError("No loci could be generated. Check FASTA file content and locus_length.")

    return pl.DataFrame(loci_list, schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "name": pl.Utf8})

def read_bed_file(bed_file_path: str) -> pl.DataFrame:
    """Reads a BED file into a Polars DataFrame."""
    try:
        df = pl.read_csv(bed_file_path, separator='\t', has_header=False, new_columns=["chrom", "start", "end", "name"])
        # Ensure correct dtypes, especially if BED file has fewer than 4 columns or name is numeric
        df = df.with_columns([
            pl.col("chrom").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("name").cast(pl.Utf8) # Cast name to string, handling potential missing or numeric names
        ])
    except Exception as e: # Broad exception for now, can be more specific
        # Fallback for 3-column BED
        try:
            df = pl.read_csv(bed_file_path, separator='\t', has_header=False, new_columns=["chrom", "start", "end"])
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("name"))
            df = df.with_columns([
                pl.col("chrom").cast(pl.Utf8),
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64)
            ])
            # Add default names if 3-column BED
            df = df.with_row_count(name="row_nr").with_columns(
                pl.col("row_nr").map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8).alias("name")
            ).drop("row_nr")

        except Exception as e_fallback:
            raise ValueError(f"Error reading BED file {bed_file_path}. Tried 4 and 3 column formats. Original error: {e}, Fallback error: {e_fallback}")

    # Ensure 'name' column exists if it was a 3-column BED or had issues
    if "name" not in df.columns:
         df = df.with_row_count(name="row_nr").with_columns(
            pl.col("row_nr").map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8).alias("name")
        ).drop("row_nr")
    elif df["name"].is_null().all(): # If name column exists but all null (e.g. from 3-col bed)
        df = df.drop("name").with_row_count(name="row_nr").with_columns(
            pl.col("row_nr").map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8).alias("name")
        ).drop("row_nr")

    return df

def write_bed_file(df: pl.DataFrame, bed_file_path: str):
    """Writes a Polars DataFrame to a BED file."""
    # Select and order columns for standard BED output
    cols_to_write = ["chrom", "start", "end"]
    if "name" in df.columns and df["name"].is_not_null().any(): # Only include name if it has non-null values
        cols_to_write.append("name")

    df.select(cols_to_write).write_csv(bed_file_path, separator='\t', include_header=False)

def main():
    parser = argparse.ArgumentParser(description="Create k-fold cross-validation splits for genomic loci.")

    # Group for specifying loci source
    loci_source_group = parser.add_mutually_exclusive_group(required=True)
    loci_source_group.add_argument("--loci_bed_file", type=str, help="Path to the input BED file containing genomic loci.")
    loci_source_group.add_argument("--fasta_file", type=str, help="Path to the genomic FASTA file, used for generating random loci if --loci_bed_file is not provided.")

    # Arguments specific to random loci generation (only relevant if --fasta_file is used as the source)
    parser.add_argument("--num_random_loci", type=int, default=10000, help="Number of random loci to generate (default: 10000). Only used if --fasta_file is the chosen loci source.")
    parser.add_argument("--locus_length", type=int, default=4096, help="Length of each random locus (default: 4096). Only used if --fasta_file is the chosen loci source.")

    parser.add_argument("--k_folds", type=int, required=True, help="The number of folds.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output BED files for each fold.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")

    args = parser.parse_args()

    # Argument validation
    if not args.loci_bed_file and not args.fasta_file:
        parser.error("Either --loci_bed_file or --fasta_file must be provided.")
    if args.loci_bed_file and args.fasta_file:
        print("Warning: --loci_bed_file is provided, so --fasta_file and related arguments for random loci generation will be ignored.")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.loci_bed_file:
        print(f"Reading loci from {args.loci_bed_file}...")
        loci_df = read_bed_file(args.loci_bed_file)
    else:
        print(f"Generating {args.num_random_loci} random loci of length {args.locus_length} from {args.fasta_file}...")
        loci_df = generate_random_loci(args.fasta_file, args.num_random_loci, args.locus_length, args.seed)

    if loci_df.is_empty():
        print("Error: No loci found or generated. Exiting.")
        return

    print(f"Successfully loaded/generated {len(loci_df)} loci.")

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Ensure indices are plain Python list for KFold
    indices = list(range(len(loci_df)))

    for i, (train_idx, test_idx) in enumerate(kf.split(indices)):
        fold_num = i + 1
        print(f"Processing fold {fold_num}/{args.k_folds}...")

        # Convert numpy arrays of indices to lists for polars indexing
        train_indices_list = train_idx.tolist()
        test_indices_list = test_idx.tolist()

        train_df = loci_df[train_indices_list]
        test_df = loci_df[test_indices_list]

        train_file_path = output_path / f"fold_{fold_num}_train_loci.bed"
        test_file_path = output_path / f"fold_{fold_num}_test_loci.bed"

        write_bed_file(train_df, str(train_file_path))
        write_bed_file(test_df, str(test_file_path))

        print(f"  Saved training loci to {train_file_path} ({len(train_df)} regions)")
        print(f"  Saved testing loci to {test_file_path} ({len(test_df)} regions)")

    print("K-fold cross-validation splits created successfully.")

if __name__ == "__main__":
    main()
