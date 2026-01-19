# GLR-BiMDA: A Global Multi-Relational Bilinear Framework for miRNA-Disease Association Type Prediction

A unified deep learning framework integrating global relational graph convolution with local attention mechanisms for predicting miRNA-disease association types, featuring bidirectional cross-attention, bilinear dual-channel decoding, and reliable negative sample selection.

## Highlights

- **Global-Local Semantic Modeling**: Integrates bidirectional cross-attention with relational graph convolution to capture both global topological features and local semantic relationships between miRNAs and diseases.
- **Bilinear Dual-Channel Decoder**: Fuses global and local features via element-level feature fusion, synergistically modeling linear and non-linear feature interactions through dual channels and attention mechanisms.
- **Reliable Negative Sample Selection**: Proposes an embedding similarity-based strategy to iteratively compare multi-view similarities between unlabeled samples and positive/negative centroids, addressing class imbalance and improving training efficiency.
- **Multi-Relational Graph Learning**: Models diverse interaction types between miRNA-miRNA, disease-disease, and miRNA-disease pairs in a unified heterogeneous graph.
- **Multi-Type Association Prediction**: Classifies associations into three distinct types: upregulation , downregulation , and other interactions .

## Framework Overview

The GLR-BiMDA framework comprises three integrated modules:

### 1. Global Multi-Relational Graph Convolution Network (G-MRGCN)
- **Bidirectional Cross-Attention**: Enhances semantic alignment between miRNA and disease embeddings through mutual attention mechanisms.
- **Relational Graph Convolution**: Captures multi-hop topological dependencies in heterogeneous miRNA-disease graphs with multiple relation types.
- **Global Semantic Aggregation**: Integrates features from both miRNA and disease views via attention-based pooling.

### 2. Bilinear Dual-Channel Decoder (BiDCD)
- **Element-Level Feature Fusion**: Combines global topological features with local semantic features at the element level.
- **Dual-Channel Interaction**: Models both linear and non-linear feature interactions through parallel processing channels.
- **Attention-Guided Fusion**: Uses multi-head attention to dynamically weight different feature sources.

### 3. Reliable Negative Sample Selection (RNSS)
- **Embedding Similarity-Based Selection**: Iteratively compares unlabeled samples with positive and negative centroids in embedding space.
- **Multi-View Similarity Assessment**: Evaluates samples from both sequence and functional perspectives.
- **Adaptive Filtering**: Dynamically adjusts selection thresholds based on training progress and class distribution.

## Architecture

The model integrates six technical innovations:

1. **Dual-View Feature Extraction**
   - miRNA sequence similarity (m_ss)
   - miRNA functional similarity (mi_fun)  
   - Disease gene similarity (d_gs)
   - Disease semantic similarity (dis_sem)

2. **Multi-Relational Graph Construction**
   - miRNA-miRNA similarity relations 
   - Disease-disease similarity relations
   - miRNA-disease association relations 

3. **Global Graph Learning**
   - Relational Graph Convolutional Networks (RGCN)
   - Bidirectional cross-attention between miRNA and disease
   - Multi-layer feature propagation with residual connections

4. **Local Attention Mechanism**
   - Neighbor attention for local structure modeling
   - Multi-head attention for feature fusion
   - Subgraph sampling with adaptive neighborhood selection

5. **Bilinear Dual-Channel Decoding**
   - Channel 1: Linear feature interactions
   - Channel 2: Non-linear feature transformations
   - Element-level fusion with attention weighting

6. **Adaptive Negative Sampling**
   - Internal negative sample selection
   - Embedding similarity-based filtering
   - Iterative centroid updating

## Requirements

```
python>=3.8
torch>=2.1.2
torch-geometric>=2.0.0
dgl>=1.1.3
numpy>=1.24.0
pandas>=1.2.0
scikit-learn>=1.3.0
scipy>=1.13.1
matplotlib>=3.7.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/GLR-BiMDA.git
cd GLR-BiMDA
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric dgl-cu121
pip install -r requirements.txt
```

## Dataset Preparation

### Dataset Structure

Place your data in the `dataset/` directory with the following files:

```
dataset/
├── m_fs.csv              # miRNA functional similarity
├── m_ss.csv              # miRNA sequence similarity
├── m_gs.csv              # miRNA Gaussian similarity
├── d_ts.csv              # Disease functional similarity
├── d_ss.csv              # Disease semantic similarity
├── d_gs.csv              # Disease Gaussian similarity
├── m_d.csv               # miRNA-disease association matrix 
└── m_d_edge.csv          # miRNA-disease association types 
```

### Data Format

- **Similarity matrices**: CSV format (NxN matrices)
- **Association matrix**: CSV format with binary values (0/1)
- **Association types**: CSV format with values -1 (downregulation), 1 (upregulation), 2 (other)

### Dataset Statistics (Example)
- **miRNAs**: 853
- **Diseases**: 591  
- **Binary Associations**: ~12,000 positive pairs
- **Association Types**: 3 (Downregulation, Upregulation, Other)

## Usage

### Quick Start

1. **Data preprocessing**:
```bash
python check_dataset.py
```

2. **Train the model** (binary classification):
```bash
python bitrain.py --epoch 50 --lr 0.0005 --batchSize 64
```

3. **Train the model** (ternary classification):
```bash
python multrain.py --epoch 50 --lr 0.0005 --batchSize 64
```

### Training Configuration

Modify parameters via command-line arguments:

```bash
python multrain.py \
    --epoch 100 \
    --lr 0.001 \
    --batchSize 128 \
    --kfold 5 \
    --nei_size 256 32 \
    --hop 2 \
    --feture_size 256 \
    --Dropout 0.2 \
    --weight_decay 0.0005
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epoch` | 50 | Number of training epochs |
| `lr` | 0.0005 | Learning rate |
| `batchSize` | 64 | Batch size |
| `kfold` | 5 | Cross-validation folds |
| `nei_size` | [256, 32] | Neighborhood sampling sizes |
| `hop` | 2 | Graph propagation hops |
| `feture_size` | 256 | Node feature dimension |
| `Dropout` | 0.1 | Dropout rate |
| `weight_decay` | 0.0005 | L2 regularization |

## Model Components

### 1. NewRGCNModel (`new_model.py`)
The core model integrating all innovations:
- `InternalNegativeSelector`: Reliable negative sample selection
- `EnhancedDualChannelDecoder`: Bilinear dual-channel decoding
- `TrueHeterogeneousRGCN`: Multi-relational graph convolution
- `SimplifiedElementFusion`: Element-level feature fusion

### 2. Graph Construction (`exSubGraph.py`)
- Subgraph sampling with multi-hop neighbors
- Adaptive neighborhood selection
- Relation-aware edge construction

### 3. Attention Layers (`otherlayers.py`)
- `CrossAttentionLayer`: Bidirectional cross-attention
- `NeiAttention`: Neighborhood attention
- `SimAttention`: Similarity attention weighting

## Results

### Performance Metrics

The model is evaluated using:

**Binary Classification:**
- AUC (Area Under ROC Curve)
- AUPR (Area Under Precision-Recall Curve)
- Accuracy, Precision, Recall, F1-Score

**Ternary Classification:**
- Top-1 Accuracy
- Weighted Precision, Recall, F1-Score
- Per-class metrics for Downregulation/Upregulation/Other

### Expected Performance

| Task | Metric | 5-Fold CV | Independent Test |
|------|--------|-----------|------------------|
| Binary | AUC | ~0.95 | ~0.92 |
| Binary | AUPR | ~0.90 | ~0.88 |
| Ternary | Top-1 Acc | ~0.75 | ~0.72 |
| Ternary | Weighted F1 | ~0.70 | ~0.68 |

## Project Structure

```
GLR-BiMDA/
├── bitrain.py                     # Binary classification training
├── multrain.py                    # Ternary classification training
├── new_model.py                   # Main model architecture (6 innovations)
├── extractSubGraph.py            # Subgraph construction
├── otherlayers.py                # Attention and graph layers
├── utils.py                      # Data loading and preprocessing
├── dataset/                      # Dataset directory
│   ├── m_fs.csv
│   ├── m_ss.csv
│   ├── m_gs.csv
│   ├── d_ts.csv
│   ├── d_ss.csv
│   ├── d_gs.csv
│   ├── m_d.csv
│   └── m_d_edge.csv
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Key Innovations Explained

### 1. Global-Local Semantic Modeling

By integrating bidirectional cross-attention with relational graph convolution, the model simultaneously captures:
- **Global topological patterns** through multi-hop graph propagation
- **Local semantic relationships** via attention-based feature alignment
- **Multi-relational interactions** across different association types

### 2. Bilinear Dual-Channel Decoder

The decoder employs two parallel channels:
- **Channel 1**: Linear transformations preserving original feature semantics
- **Channel 2**: Non-linear transformations capturing complex interactions
- **Element-Level Fusion**: Combines features at the finest granularity with attention-based weighting

### 3. Reliable Negative Sample Selection

Addressing class imbalance through:
- **Iterative Centroid Updating**: Dynamically adjusts positive/negative centroids
- **Multi-View Similarity**: Considers both sequence and functional perspectives
- **Adaptive Filtering**: Selects high-confidence negative samples based on embedding similarities
