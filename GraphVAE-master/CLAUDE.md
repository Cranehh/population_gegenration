# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of GraphVAE for molecular graph generation, based on the paper by Simonovsky & Komodakis (2018). The implementation focuses specifically on generating small molecular graphs using variational autoencoders.

## Environment Setup

Python environment managed via conda:
- Python 3.6, PyTorch 1.1
- Environment file: `conda_environment.yml` (GPU) or `conda_environment_no_gpu.yml`
- Setup dependencies with submodules: `git submodule init && git submodule update`
- Configure Python path: `source set_up.sh`

## Key Commands

### Environment Setup
```bash
conda env create -f conda_environment.yml
source set_up.sh  # Sets PYTHONPATH for submodules
```

### Training
```bash
python scripts/train_graphvae.py <dataset_name> [--mpm] [--max_num_nodes=<num>]
# Example: python scripts/train_graphvae.py qm9 --mpm --max_num_nodes=9
```

### Sampling
```bash
python scripts/sample_graphvae.py <weights_name> [--num_samples=<num>]
# Example: python scripts/sample_graphvae.py scripts/saved_weights_and_samples/qm9/gvae_weights_qm9.pth.pick
```

### Testing
```bash
pytest tests/
```

### Docker
```bash
docker build -t johnbradshaw/graph-vae .
docker run -it johnbradshaw/graph-vae:latest
```

## Architecture

### Core Modules
- `graph_vae/`: Main package containing VAE implementation
  - `graph_vae_model.py`: Core VAE model architecture
  - `graph_datastructure.py`: Graph representation and manipulation utilities
  - `smiles_data.py`: SMILES string processing and molecular data handling
- `submodules/`: External dependencies
  - `GNN/`: Graph Neural Network implementations
  - `autoencoders/`: Autoencoder utilities

### Data Flow
1. SMILES strings → Graph representations (adjacency matrices + node features)
2. Graph encoder → Latent space representation
3. Latent space → Graph decoder → Generated graphs
4. Optional graph matching (MPM) for improved training

### Datasets
- QM9: Small organic molecules (`qm9_smiles.txt`)
- ZINC: Drug-like molecules (`zinc_leq20nodes.txt`)

## Code Notation
- `b`: batch size
- `e`: number of edge types  
- `v`: number of nodes in adjacency matrix
- `v*`: number of stacked active nodes across all graphs
- `E*`: number of edges across all graphs
- `h`: dimension of node representation
- `g`: number of groups

## Training Configuration
Default hyperparameters in `train_graphvae.py`:
- Latent dimension: 40
- Beta (KL weight): 1/40
- Epochs: 25
- Batch size: 32
- Learning rate: 1e-3