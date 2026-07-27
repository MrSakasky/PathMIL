# PathMIL

<p align="center">
  <strong>PathMIL: Path-aware Multiple Instance Learning for Weakly Supervised Whole-Slide Image Classification</strong>
</p>

<p align="center">
  Official implementation of PathMIL, a path-aware dual-stream multiple instance learning framework for weakly supervised whole-slide image classification.
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework">Framework</a> •
  <a href="#installation">Installation</a> •
  <a href="#data-preparation">Data Preparation</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#visualization">Visualization</a> •
  <a href="#citation">Citation</a>
</p>

---

## News

- **Paper link will be updated after publication.**

## Overview

Multiple instance learning enables whole-slide image classification using only slide-level annotations. Most existing MIL methods represent a whole-slide image as an unordered collection of patches and directly aggregate patch-level evidence for slide-level prediction. However, this formulation provides limited modeling of the continuous and directional organization of diagnostically relevant morphology across neighboring tissue regions.

PathMIL introduces an intermediate **path-level representation** between patch encoding and slide aggregation. Its core component, **Dynamic Pathway Modeling (DPM)**, constructs short directed local pathways from individual seed patches. At each step, neighboring candidates are evaluated by combining local feature similarity, spatial bias, and global guidance.

The resulting pathway features are aggregated into path-level embeddings and fused with patch-level representations. A **Gated Multi-scale Positional Encoding module (G-PPEG)** further captures local tissue continuity and hierarchical spatial regularities before slide-level MIL aggregation.

PathMIL therefore models both:

- patch-level morphological appearance;
- path-level directional and spatial organization.

Experiments on four public benchmarks and one internal neuropathology cohort demonstrate consistent improvements across diverse whole-slide image classification tasks.

<p align="center">
  <img src="assets/framework.png" width="95%" alt="PathMIL framework">
</p>


## Highlights

- **Path-level Evidence Modeling**  
  Introduces an intermediate representation between patch-level features and slide-level prediction.

- **Dynamic Pathway Modeling**  
  Constructs short directed local pathways by integrating local feature similarity, spatial relationships, and global guidance.

- **Dual-stream Representation Learning**  
  Jointly models patch-level appearance and path-level morphological organization.

- **Gated Multi-scale Positional Encoding**  
  Adaptively captures tissue continuity and hierarchical spatial regularities.

- **Pathway-based Visualization**  
  Complements conventional patch-level attention heatmaps with spatially coherent local trajectories.

## Framework

PathMIL contains four major stages.

### 1. Patch Feature Encoding

A pretrained feature encoder extracts patch embeddings from tissue regions in a whole-slide image:

```text
Patch features: [N, C]
Coordinates:    [N, 2]
```

where:

- `N` is the number of tissue patches;
- `C` is the feature dimension.

PathMIL is not restricted to a specific feature encoder. CNN-based, Transformer-based, and pathology foundation model-based encoders can be used.

### 2. Dynamic Pathway Modeling

For each seed patch, DPM retrieves local candidate neighbors and recursively selects the next pathway node.

The transition score combines:

- local feature similarity;
- spatial bias;
- global guidance.

A short directed local pathway is constructed for each patch. Intra-path attention then aggregates the selected pathway features into a path-level representation.

### 3. Patch-Path Fusion and G-PPEG

Patch-level and path-level representations are projected into a shared feature space and fused.

G-PPEG reshapes the fused instances into a pseudo-grid and applies multi-scale depthwise convolutions. A token-wise gate controls the contribution of positional information to each instance representation.

### 4. Slide-level Aggregation

Gated-attention MIL pooling aggregates the refined instance representations into a slide-level embedding for classification.

> The learned pathways represent the model's local organization of morphological evidence. They should not be interpreted as literal biological progression routes or as pathologists' reading sequences.

## Repository Structure

The repository is organized as follows:

```text
PathMIL/
├── assets/
│   └── pathmil_framework.png
├── configs/
│   ├── camelyon16.yaml
│   ├── tcga_lung.yaml
│   ├── tcga_nsclc.yaml
│   └── panda.yaml
├── datasets/
│   ├── dataset_generic.py
│   └── dataset_utils.py
├── models/
│   ├── pathmil.py
│   ├── dpm.py
│   ├── gppeg.py
│   └── attention.py
├── scripts/
│   ├── run_train.sh
│   ├── run_eval.sh
│   └── run_visualization.sh
├── splits/
├── train.py
├── eval.py
├── visualize_paths.py
├── environment.yml
├── requirements.txt
├── LICENSE
└── README.md
```

The exact filenames may be adjusted in the final release.

## Installation

### Clone the repository

```bash
git clone https://github.com/MrSakasky/PathMIL.git
cd PathMIL
```

### Create the environment

We recommend using Conda:

```bash
conda env create -f environment.yml
conda activate pathmil
```

Alternatively, install the required packages with:

```bash
pip install -r requirements.txt
```

### Main dependencies

The main dependencies include:

```text
Python
PyTorch
torchvision
OpenSlide
h5py
NumPy
pandas
scikit-learn
matplotlib
PyYAML
tqdm
```

The exact package versions used in our experiments will be provided in `environment.yml`.

## Data Preparation

PathMIL performs MIL training using pre-extracted patch features and their spatial coordinates.

A standard preprocessing pipeline contains the following steps:

1. Segment tissue regions from each whole-slide image.
2. Crop image patches from tissue regions.
3. Extract patch features using a pretrained encoder.
4. Save patch features and spatial coordinates.
5. Prepare slide-level labels and cross-validation splits.

### Recommended directory structure

```text
DATA_ROOT/
├── pt_files/
│   ├── slide_001.pt
│   ├── slide_002.pt
│   └── ...
├── h5_files/
│   ├── slide_001.h5
│   ├── slide_002.h5
│   └── ...
├── labels.csv
└── splits/
    ├── splits_0.csv
    ├── splits_1.csv
    └── ...
```

Each slide should contain patch features and corresponding coordinates:

```text
features: [N, C]
coords:   [N, 2]
```

The ordering of `coords` must be consistent with the ordering of `features`.

### Label file

A minimal label file can be organized as follows:

```csv
slide_id,label
slide_001,0
slide_002,1
slide_003,0
```

For multi-class tasks:

```csv
slide_id,label
slide_001,0
slide_002,1
slide_003,2
```

### Feature encoders

PathMIL can use features extracted by different pretrained encoders, including:

- ImageNet-pretrained CNNs;
- pathology-specific CNNs;
- vision Transformers;
- pathology foundation models.

Please ensure that the feature dimension in the configuration file matches the dimension of the extracted features.

## Configuration

Dataset-specific settings are stored in YAML configuration files.

An example configuration is shown below:

```yaml
dataset:
  name: camelyon16
  data_root: /path/to/CAMELYON16/features
  label_csv: /path/to/CAMELYON16/labels.csv
  split_dir: splits/camelyon16
  num_classes: 2

model:
  input_dim: 1024
  hidden_dim: 512
  attention_dim: 256

  path_depth: 3
  num_neighbors: 16

  alpha: 1.0
  beta: 0.5
  gamma: 0.5

  dropout: 0.25
  regularization_lambda: 0.01

training:
  epochs: 200
  learning_rate: 0.0002
  weight_decay: 0.00001
  bag_weight: 0.7
  seed: 1

output:
  save_dir: results/camelyon16
```

## Training

Train PathMIL on one fold:

```bash
python train.py \
  --config configs/camelyon16.yaml \
  --fold 0
```

## Evaluation

Evaluate a trained checkpoint using:

```bash
python eval.py \
  --config configs/camelyon16.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --fold 0
```

The evaluation script reports slide-level classification metrics such as:

- area under the ROC curve;
- accuracy;
- precision;
- recall;
- F1 score.

The exact reported metrics may vary according to the task.

## Visualization

PathMIL supports visualization of patch-level attention and learned local pathways.

```bash
python visualize_paths.py \
  --config configs/camelyon16.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --slide_id slide_001 \
  --output_dir results/visualization
```

The visualization module can be used to inspect pathway behaviors such as:

- coherent pathways within morphologically consistent regions;
- boundary-aware exploration between neighboring tissue patterns;
- local interactions across heterogeneous tissue regions.

The learned pathways should be interpreted together with the original histology and patch-level attention maps.

## Datasets

We evaluate PathMIL on four public datasets and one internal neuropathology cohort.

| Dataset | Task | Availability |
|---|---|---|
| CAMELYON16 | Lymph-node metastasis classification | Public |
| TCGA-Lung | LUAD versus LUSC classification | Public |
| TCGA-NSCLC | Pathological T-stage-related classification | Public |
| PANDA | Prostate cancer ISUP grade prediction | Public |
| Brain Bank | Aβ plaque micro-lesion pattern classification | Internal |

Public datasets should be downloaded from their official sources. Users must follow the corresponding licenses and data-use agreements.

The internal Brain Bank cohort cannot be redistributed through this repository because it remains subject to institutional ethics and data-governance requirements.



## Acknowledgements

We thank the developers of the open-source computational pathology and multiple instance learning projects that support this work.[CLAM](https://github.com/mahmoodlab/CLAM)

Human brain tissue used in this study is provided by the National Human Brain Bank for Development and Function, Chinese Academy of Medical Sciences and Peking Union Medical College, with support from the relevant brain banking and neuroscience resources.

The use of the internal cohort follows the institutional ethics approvals and data-governance requirements described in the manuscript.



## Contact
For questions, suggestions, or bug reports, please open an issue in this repository.

