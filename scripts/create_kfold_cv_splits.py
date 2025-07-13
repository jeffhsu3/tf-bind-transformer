import argparse
import polars as pl
from pyfaidx import Fasta
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path
import random
import yaml
from intervaltree import Interval, IntervalTree


def generate_random_loci(
    fasta_file: str, num_loci: int, locus_length: int, seed: int
) -> pl.DataFrame:
    """
    Generates random, non-overlapping genomic loci from a FASTA file.

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
    interval_trees = {chrom: IntervalTree() for chrom in fasta.keys()}

    chrom_keys = sorted(list(fasta.keys()))  # Ensure consistent order

    max_attempts_per_locus = 1000  # Increased attempts for non-overlapping constraint

    loci_generated_count = 0
    while loci_generated_count < num_loci:
        chrom_name = random.choice(chrom_keys)
        chromosome = fasta[chrom_name]
        chrom_len = len(chromosome)

        if chrom_len < locus_length:
            continue

        for _ in range(max_attempts_per_locus):
            start = np.random.randint(0, chrom_len - locus_length + 1)
            end = start + locus_length

            # Check for overlaps
            if not interval_trees[chrom_name].overlaps(start, end):
                interval_trees[chrom_name].add(Interval(start, end))
                loci_list.append(
                    {
                        "chrom": chrom_name,
                        "start": start,
                        "end": end,
                        "name": f"random_locus_{loci_generated_count + 1}",
                    }
                )
                loci_generated_count += 1
                break  # Move to the next locus
        else:  # This else corresponds to the for loop
            print(
                f"Warning: Could not generate non-overlapping locus after {max_attempts_per_locus} attempts. "
                f"Generated {loci_generated_count}/{num_loci} loci so far. "
                "Consider reducing num_loci, locus_length, or checking chromosome sizes."
            )
            # Decide if to continue or stop. For now, let's stop to avoid long runtimes.
            break

    if not loci_list:
        raise ValueError(
            "No loci could be generated. Check FASTA file content and locus_length."
        )

    if loci_generated_count < num_loci:
        print(f"Warning: Only generated {loci_generated_count} out of {num_loci} requested loci.")

    return pl.DataFrame(
        loci_list,
        schema={"chrom": pl.Utf8, "start": pl.Int64, "end": pl.Int64, "name": pl.Utf8},
    )


def read_bed_file(bed_file_path: str) -> pl.DataFrame:
    """Reads a BED file into a Polars DataFrame."""
    try:
        df = pl.read_csv(
            bed_file_path,
            separator="\t",
            has_header=False,
            new_columns=["chrom", "start", "end", "name"],
        )
        # Ensure correct dtypes, especially if BED file has fewer than 4 columns or name is numeric
        df = df.with_columns(
            [
                pl.col("chrom").cast(pl.Utf8),
                pl.col("start").cast(pl.Int64),
                pl.col("end").cast(pl.Int64),
                pl.col("name").cast(
                    pl.Utf8
                ),  # Cast name to string, handling potential missing or numeric names
            ]
        )
    except Exception as e:  # Broad exception for now, can be more specific
        try:
            df = pl.read_csv(
                bed_file_path,
                separator="\t",
                has_header=False,
                new_columns=["chrom", "start", "end"],
            )
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias("name"))
            df = df.with_columns(
                [
                    pl.col("chrom").cast(pl.Utf8),
                    pl.col("start").cast(pl.Int64),
                    pl.col("end").cast(pl.Int64),
                ]
            )
            # Add default names if 3-column BED
            df = (
                df.with_row_count(name="row_nr")
                .with_columns(
                    pl.col("row_nr")
                    .map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8)
                    .alias("name")
                )
                .drop("row_nr")
            )

        except Exception as e_fallback:
            raise ValueError(
                f"Error reading BED file {bed_file_path}. Tried 4 and 3 column formats. Original error: {e}, Fallback error: {e_fallback}"
            )

    # Ensure 'name' column exists if it was a 3-column BED or had issues
    if "name" not in df.columns:
        df = (
            df.with_row_count(name="row_nr")
            .with_columns(
                pl.col("row_nr")
                .map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8)
                .alias("name")
            )
            .drop("row_nr")
        )
    elif (
        df["name"].is_null().all()
    ):  # If name column exists but all null (e.g. from 3-col bed)
        df = (
            df.drop("name")
            .with_row_count(name="row_nr")
            .with_columns(
                pl.col("row_nr")
                .map_elements(lambda x: f"locus_{x+1}", return_dtype=pl.Utf8)
                .alias("name")
            )
            .drop("row_nr")
        )

    return df


def write_bed_file(df: pl.DataFrame, bed_file_path: str):
    """Writes a Polars DataFrame to a BED file."""
    # Select and order columns for standard BED output
    cols_to_write = ["chrom", "start", "end"]
    if (
        "name" in df.columns and df["name"].is_not_null().any()
    ):  # Only include name if it has non-null values
        cols_to_write.append("name")

    df.select(cols_to_write).write_csv(
        bed_file_path, separator="\t", include_header=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create k-fold cross-validation splits for genomic loci."
    )

    parser.add_argument(
        "--genome_config",
        type=str,
        help="Path to a YAML config file. Can provide 'genome_fasta', 'num_random_loci', and 'locus_length'.",
    )

    # Group for specifying loci source
    loci_source_group = parser.add_mutually_exclusive_group()
    loci_source_group.add_argument(
        "--loci_bed_file",
        type=str,
        help="Path to the input BED file containing genomic loci.",
    )
    loci_source_group.add_argument(
        "--fasta_file",
        type=str,
        help="Path to the genomic FASTA file, used for generating random loci if --loci_bed_file is not provided.",
    )

    # Arguments specific to random loci generation (only relevant if --fasta_file is used as the source)
    parser.add_argument(
        "--num_random_loci",
        type=int,
        default=None,
        help="Number of random loci to generate (default: 10000). Only used if --fasta_file is the chosen loci source.",
    )
    parser.add_argument(
        "--locus_length",
        type=int,
        default=None,
        help="Length of each random locus (default: 4096). Only used if --fasta_file is the chosen loci source.",
    )

    parser.add_argument(
        "--k_folds", type=int, required=True, help="The number of folds."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output BED files for each fold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--validation_split_ratio",
        type=float,
        default=0.0,
        help="Proportion of training data to use for validation (e.g., 0.1). Defaults to 0.0",
    )

    args = parser.parse_args()

    # Load from YAML if provided and update args
    if args.genome_config:
        with open(args.genome_config, "r") as f:
            config = yaml.safe_load(f)
        if "genome_fasta" in config:
            if args.fasta_file is None:
                args.fasta_file = config["genome_fasta"]
            else:
                print(
                    f"Warning: --fasta_file is provided, so 'genome_fasta' from {args.genome_config} will be ignored."
                )
        if "num_random_loci" in config and args.num_random_loci is None:
            args.num_random_loci = int(config["num_random_loci"])

        if "locus_length" in config and args.locus_length is None:
            args.locus_length = int(config["locus_length"])

    # Set defaults for optional args if not provided
    if args.num_random_loci is None:
        args.num_random_loci = 10000
    if args.locus_length is None:
        args.locus_length = 4096

    # Argument validation
    if not args.loci_bed_file and not args.fasta_file:
        parser.error(
            "A source for loci must be provided via --loci_bed_file, --fasta_file, or --genome_config (with a 'genome_fasta' key)."
        )
    if args.loci_bed_file and args.fasta_file:
        print(
            "Warning: --loci_bed_file is provided, so --fasta_file and related arguments for random loci generation will be ignored."
        )
    if not (0.0 <= args.validation_split_ratio < 1.0):
        parser.error("--validation_split_ratio must be between 0.0 and 1.0 (exclusive of 1.0).")


    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.loci_bed_file:
        print(f"Reading loci from {args.loci_bed_file}...")
        loci_df = read_bed_file(args.loci_bed_file)
    else:
        print(
            f"Generating {args.num_random_loci} random loci of length {args.locus_length} from {args.fasta_file}..."
        )
        loci_df = generate_random_loci(
            args.fasta_file, args.num_random_loci, args.locus_length, args.seed
        )

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

        if args.validation_split_ratio > 0.0:
            # Shuffle train_df before splitting to ensure randomness
            # Note: KFold already shuffles the initial dataset if shuffle=True,
            # but we might want an independent shuffle for the train/validation split
            # or rely on KFold's initial shuffle. For simplicity here, let's
            # assume KFold's shuffle is sufficient for now, or one could re-shuffle train_df.
            # Re-shuffling train_df before splitting for validation:
            train_df = train_df.sample(fraction=1.0, shuffle=True, seed=args.seed + fold_num) # ensure different seed per fold for this shuffle

            num_validation_samples = int(len(train_df) * args.validation_split_ratio)
            if num_validation_samples == 0 and len(train_df) > 0 and args.validation_split_ratio > 0:
                # Ensure at least one sample for validation if ratio is non-zero and data exists
                num_validation_samples = 1

            if num_validation_samples > 0 and num_validation_samples < len(train_df):
                validation_df = train_df.slice(0, num_validation_samples)
                train_df = train_df.slice(num_validation_samples, len(train_df) - num_validation_samples)

                validation_file_path = output_path / f"fold_{fold_num}_validation_loci.bed"
                write_bed_file(validation_df, str(validation_file_path))
                print(f"  Saved validation loci to {validation_file_path} ({len(validation_df)} regions)")
            else:
                # Not enough data to create a validation set, or ratio is too small
                # Keep all data for training in this case
                print(f"  Warning: Not enough data or validation_split_ratio too small for fold {fold_num}. Skipping validation split for this fold. All {len(train_df)} samples used for training.")
                validation_df = pl.DataFrame() # Empty dataframe

        train_file_path = output_path / f"fold_{fold_num}_train_loci.bed"
        test_file_path = output_path / f"fold_{fold_num}_test_loci.bed"

        write_bed_file(train_df, str(train_file_path))
        write_bed_file(test_df, str(test_file_path))

        print(f"  Saved training loci to {train_file_path} ({len(train_df)} regions)")
        print(f"  Saved testing loci to {test_file_path} ({len(test_df)} regions)")


    print("K-fold cross-validation splits created successfully.")


if __name__ == "__main__":
    main()
