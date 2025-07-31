# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
```bash
# Install package in development mode
pip install -e .

# Or install from PyPI (when available)
pip install tf-bind-transformer

# Install development dependencies (optional)
pip install -e ".[dev]"

# Install additional conda packages (if using conda)
conda install --channel conda-forge --channel bioconda pybedtools pyBigWig

# Set up pre-commit hooks (for development)
pre-commit install
```

### Testing
```bash
# Run all tests using unittest
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_data_bigwig

# Alternative: Run tests with pytest (compatible)
pytest tests/ -v
```

### Code Quality
```bash
# Run linting and formatting with ruff
ruff check .
ruff format .

# Run pre-commit hooks manually
pre-commit run --all-files
```

### K-Fold Cross-Validation
```bash
# Generate k-fold CV splits from existing BED file
python scripts/create_kfold_cv_splits.py \
    --loci_bed_file path/to/loci.bed \
    --k_folds 5 \
    --output_dir cv_splits/ \
    --seed 42

# Generate k-fold CV splits from random loci
python scripts/create_kfold_cv_splits.py \
    --fasta_file path/to/genome.fna \
    --num_random_loci 10000 \
    --k_folds 5 \
    --output_dir cv_splits/
```

## Architecture Overview

### Core Components

The tf-bind-transformer is a PyTorch-based framework for predicting transcription factor binding using transformers and attention mechanisms. The architecture consists of:

1. **AdapterModel** (`tf_bind_transformer/tf_bind_transformer.py`):
   - Main model class wrapping Enformer with additional adapter layers
   - Handles both binary prediction (bind/no-bind) and continuous track prediction
   - Supports protein embedding integration via ESM or ProtAlbert
   - Implements contextual conditioning through FiLM, squeeze-excite, or hypergrid methods

2. **Enformer Integration**:
   - Uses Enformer as the base genomic sequence encoder
   - Supports fine-tuning of Enformer layers (full, layer-norm only, or frozen)
   - Caches genetic sequence embeddings for efficiency during training

3. **Protein Embedding System** (`tf_bind_transformer/protein_utils.py`):
   - Configurable protein encoders (ESM, ProtAlbert)
   - Caching system for protein embeddings to avoid recomputation
   - Supports both single proteins and protein complexes

4. **Contextual Conditioning** (`tf_bind_transformer/context_utils.py`):
   - Free-text context encoding using PubMed-trained transformers
   - Supports cell type and experimental parameter conditioning
   - Multiple conditioning methods: FiLM, squeeze-excite, hypergrid

5. **Attention Mechanisms** (`tf_bind_transformer/attention.py`):
   - FILIP-style fine-grained interaction between DNA and protein sequences
   - Joint cross-attention blocks for protein-genome interaction
   - Self-attention blocks for genomic sequence refinement

### Data Processing Pipeline

1. **BigWig Data** (`tf_bind_transformer/data_bigwig.py`):
   - `BigWigDataset` for loading genomic tracks from BigWig files
   - `BigWigTracksOnlyDataset` for track-only data without protein information
   - Handles annotation files, genomic intervals, and protein factor mapping

2. **Peak Data** (`tf_bind_transformer/data.py`):
   - `RemapAllPeakDataset` for ChIP-seq peak data processing
   - Negative sample generation from genome-wide random regions
   - Filtering based on blacklist regions and existing peaks

3. **Protein Data**:
   - `FactorProteinDataset` for transcription factor protein sequences
   - FASTA file processing for protein complexes
   - Uniprot integration for protein sequence retrieval

### Training Framework

1. **Binary Prediction Trainer** (`tf_bind_transformer/training_utils.py`):
   - `Trainer` class for bind/no-bind prediction tasks
   - Handles chromosome-based train/validation splits
   - Supports held-out target validation

2. **Track Prediction Trainer** (`tf_bind_transformer/training_utils_bigwig.py`):
   - `BigWigTrainer` for continuous track prediction
   - Integrates with BigWig data pipeline
   - Supports multi-track training

### Key Design Patterns

1. **Caching System**: Extensive use of caching for expensive operations (protein embeddings, genomic sequences)
2. **Flexible Conditioning**: Multiple conditioning strategies can be mixed and matched
3. **Modular Architecture**: Clear separation between data processing, model components, and training
4. **Memory Optimization**: Optional CPU-based embedding computation to save GPU memory

### Environment Variables

- `CONTEXT_EMBED_USE_CPU=1`: Run context embedding on CPU
- `PROTEIN_EMBED_USE_CPU=1`: Run protein embedding on CPU  
- `VERBOSE=1`: Enable verbose caching output
- `CLEAR_CACHE=1`: Force cache clearance on startup

### File Organization

- `tf_bind_transformer/`: Main package with core model components
- `scripts/`: Data processing and utility scripts
- `tests/`: Unit tests (uses unittest framework)
- `data/`: Configuration files (experiments.json)
- Generated output directories for cross-validation splits and cached data

The codebase follows scientific computing best practices with comprehensive documentation, caching for efficiency, and modular design for different use cases (binary prediction, track prediction, protein complexes).