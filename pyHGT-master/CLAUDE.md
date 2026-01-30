# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is an implementation of Heterogeneous Graph Transformer (HGT), a graph neural network architecture for large-scale heterogeneous and dynamic graphs. The project is based on PyTorch Geometric and includes implementations for various downstream tasks on academic graph datasets.

## Architecture

The codebase is organized into three main directories, each containing identical pyHGT modules but for different use cases:

- `pyHGT/` - Core library implementation
- `OAG/` - Open Academic Graph specific experiments 
- `ogbn-mag/` - OGB MAG dataset experiments

### Core Components

- `conv.py` - Heterogeneous graph convolutional layers (HGTConv, GCNConv, GATConv, etc.)
- `model.py` - Model components including Classifier and Matcher modules
- `data.py` - Graph data structure and sampling algorithms
  - `class Graph` - Heterogeneous graph data structure with node features as pandas DataFrame and edge lists as dictionaries
  - `sample_subgraph()` - Sampling algorithm for heterogeneous graphs
- `train_*.py` - Training scripts for specific tasks (paper-field, author-disambiguation, paper-venue)
- `preprocess_*.py` - Data preprocessing scripts

### Model Types Supported

- `hgt` - Heterogeneous Graph Transformer (default)
- `gcn` - Graph Convolutional Network
- `gat` - Graph Attention Network
- `rgcn` - Relational Graph Convolutional Network
- `han` - Heterogeneous Attention Network
- `hetgnn` - Heterogeneous Graph Neural Network

## Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training Commands

Paper-field classification (L2):
```bash
python3 train_paper_field.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --conv_name hgt
```

Author disambiguation:
```bash
python3 train_author_disambiguation.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --conv_name hgt
```

Paper-venue classification:
```bash
python3 train_paper_venue.py --data_dir PATH_OF_DATASET --model_dir PATH_OF_SAVED_MODEL --conv_name hgt
```

### Key Training Parameters

- `--conv_name` - Model type (hgt, gcn, gat, rgcn, han, hetgnn)
- `--sample_depth` - Depth of sampled subgraph (default: 6)
- `--sample_width` - Number of nodes sampled per layer per type (default: 128)
- `--n_pool` - Number of parallel sampling processes (default: 4)
- `--repeat` - Number of times to reuse sampled batch (default: 2)
- `--n_hid` - Hidden dimension size (default: 400)
- `--n_heads` - Number of attention heads (default: 8)
- `--n_layers` - Number of GNN layers (default: 4)

## Data Structure

The Graph class stores:
- Node features in `Graph.node_feature` as pandas DataFrame
- Adjacency matrices in `Graph.edge_list` as nested dictionaries indexed by `<target_type, source_type, relation_type, target_id, source_id>`
- Time information for temporal graph analysis

## Dependencies

Core dependencies include:
- PyTorch 1.13.1
- torch-geometric 1.3.2
- pandas 0.24.2
- numpy 1.22.0
- transformers 4.30.0
- dill, tqdm, seaborn, matplotlib

## Performance Tuning

- Reduce `sample_depth` and `sample_width` if GPU memory is insufficient
- Increase `n_pool` for faster sampling on high-memory machines
- Increase `repeat` if training time is much smaller than sampling time