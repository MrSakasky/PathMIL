# PathMIL
Official implementation of PathMIL, a path-aware dual-stream MIL framework for weakly supervised whole-slide image classification. PathMIL introduces Dynamic Pathway Modeling to capture continuous and directional morphological evidence, and fuses patch- and path-level representations with gated multi-scale positional encoding.
PathMIL

PathMIL: Path-aware Multiple Instance Learning for Weakly Supervised Whole-Slide Image Classification

Official implementation of PathMIL, a path-aware dual-stream multiple instance learning framework that models continuous and directional morphological evidence in whole-slide images.

Paper: Coming soon
Code and pretrained models: To be released

<p align="center"> <img src="assets/pathmil_framework.png" width="95%" alt="Overview of the PathMIL framework"> </p>

📇 Overview

Multiple instance learning (MIL) enables whole-slide image (WSI) classification using only slide-level labels. Most existing methods represent a slide as an unordered collection of image patches and aggregate patch-level evidence directly for slide-level prediction. Although effective, this formulation provides limited modeling of the continuous and directional organization of diagnostically relevant morphology across neighboring tissue regions.

PathMIL introduces an intermediate path-level representation between patch encoding and slide aggregation. Its core component, Dynamic Pathway Modeling (DPM), constructs short directed local pathways from individual seed patches. At each step, candidate neighboring patches are evaluated by combining local feature similarity, spatial bias, and global guidance. The selected pathway features are then aggregated into path-level embeddings through intra-path attention.

PathMIL adopts a dual-stream design that integrates patch-level appearance with path-level morphological evidence. The fused instance representations are further refined by Gated Multi-scale Positional Encoding (G-PPEG) before gated-attention MIL pooling and slide-level classification. This design enables the model to capture local tissue continuity, directional morphological transitions, and hierarchical spatial regularities while retaining the discriminative information of individual patches.

Experiments on four public benchmarks and one internal neuropathology cohort demonstrate consistent improvements across diverse WSI classification scenarios. Pathway visualization further reveals spatially coherent local trajectories that are broadly aligned with routine pathological diagnostic reasoning for the displayed tissue regions.

✨ Highlights
Path-level evidence modeling: introduces an intermediate representation between patch features and slide-level prediction.
Dynamic Pathway Modeling: learns short directed local pathways by integrating feature similarity, spatial relationships, and global guidance.
Dual-stream representation: jointly models patch-level appearance and pathway-level morphological organization.
Gated multi-scale positional encoding: adaptively captures local and hierarchical spatial regularities.
Pathway-based visualization: complements conventional patch-level attention heatmaps with spatially coherent local trajectories.
🧠 Framework

PathMIL consists of four main stages:

Feature encoding
A pretrained encoder extracts patch embeddings from tissue regions of a WSI.
Dynamic Pathway Modeling
DPM retrieves local candidate neighbors and recursively selects pathway nodes using a learned transition score composed of:
local feature similarity;
spatial bias;
global guidance.
Patch-path fusion with G-PPEG
Patch-level and path-level embeddings are projected and fused. G-PPEG then injects gated multi-scale positional information into the fused instance representations.
Slide-level aggregation
Gated-attention MIL pooling aggregates the refined instance features for slide-level classification.

The learned pathways describe the model's local organization of morphological evidence. They should not be interpreted as literal biological progression routes or pathologists' reading sequences.

🗄️ Environment

We recommend creating a dedicated Conda environment:

conda env create -f environment.yml
conda activate pathmil

The implementation is based on Python and PyTorch. The exact package versions used for the experiments will be provided in environment.yml.

Main dependencies include:

Python
PyTorch
torchvision
OpenSlide
h5py
NumPy
pandas
scikit-learn
matplotlib
🗃️ Data Preparation

PathMIL performs MIL training on pre-extracted patch features and their spatial coordinates. A standard preprocessing pipeline contains the following steps:

segment tissue regions from each WSI;
crop non-overlapping or partially overlapping image patches;
extract patch features using a pretrained encoder;
save the patch features and coordinates using the same slide identifier;
prepare a CSV file containing slide identifiers and slide-level labels.

An example feature directory is shown below:

DATA_ROOT/
├── pt_files/
│   ├── slide_001.pt
│   ├── slide_002.pt
│   └── ...
├── h5_files/
│   ├── slide_001.h5
│   ├── slide_002.h5
│   └── ...
└── labels.csv

Each slide should contain:

features: [N, C]
coords:   [N, 2]

where N is the number of tissue patches and C is the feature dimension. The coordinates must follow the same ordering as the patch features.

A minimal label file may follow this format:

slide_id,label
slide_001,0
slide_002,1

The feature extractor is not restricted to a specific backbone. CNN-, Transformer-, and pathology foundation model-based encoders can be used as long as the extracted feature dimension is consistent with the model configuration.

🗂️ Repository Structure

The released repository will follow a structure similar to:

PathMIL/
├── assets/                 # Framework and visualization figures
├── configs/                # Dataset-specific configuration files
├── datasets/               # Dataset loaders and preprocessing utilities
├── models/
│   ├── pathmil.py          # Overall PathMIL architecture
│   ├── dpm.py              # Dynamic Pathway Modeling
│   ├── gppeg.py            # Gated multi-scale positional encoding
│   └── attention.py        # Intra-path and MIL attention modules
├── scripts/                # Training, evaluation, and visualization scripts
├── splits/                 # Cross-validation split files
├── train.py
├── eval.py
├── visualize_paths.py
├── environment.yml
└── README.md

The filenames may be adjusted in the final public release.

🚀 Training

The following command illustrates the expected training interface:

python train.py \
  --config configs/camelyon16.yaml \
  --fold 0

A configuration file should specify the dataset paths, model parameters, optimization settings, and output directory. A representative PathMIL setting is:

Parameter	Reference value
Path depth	3
Number of neighbors	16
Local score weight alpha	1.0
Spatial score weight beta	0.5
Global score weight gamma	0.5
DPM regularization coefficient	0.01
Input feature dimension	1024
Hidden dimension	512
Attention dimension	256
Learning rate	2e-4
Weight decay	1e-5
Dropout	0.25

These values provide the reference configuration used in our experiments. Dataset-specific configuration files should be used to reproduce the reported results.

For five-fold cross-validation:

for fold in 0 1 2 3 4
do
  python train.py \
    --config configs/camelyon16.yaml \
    --fold ${fold}
done
🔍 Evaluation

Evaluate a trained checkpoint with:

python eval.py \
  --config configs/camelyon16.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --fold 0

The evaluation script reports slide-level classification metrics such as accuracy, AUC, and other task-specific measures defined in the corresponding configuration.

🧭 Pathway Visualization

PathMIL supports visualization of both patch-level attention and learned local pathways:

python visualize_paths.py \
  --config configs/camelyon16.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --slide_id slide_001 \
  --output_dir results/visualization

The visualization module can be used to inspect representative pathway behaviors, including:

coherent pathways within morphologically consistent regions;
boundary-aware exploration between neighboring tissue patterns;
local interactions across heterogeneous tissue regions.

The displayed paths are model-derived relational structures and should be interpreted together with the original histology and patch-level attention maps.

📊 Datasets

We evaluate PathMIL on four public WSI benchmarks and one internal neuropathology cohort:

Dataset	Diagnostic task	Availability
CAMELYON16	Lymph-node metastasis classification	Public
TCGA-Lung	LUAD versus LUSC subtype classification	Public
TCGA-NSCLC	Pathological T-stage-related classification	Public
PANDA	Prostate cancer ISUP grade prediction	Public
Brain Bank	Aβ plaque micro-lesion pattern classification	Internal cohort

Please download the public datasets from their official sources and follow the corresponding licenses and data-use agreements. The internal Brain Bank cohort cannot be redistributed through this repository and remains subject to institutional ethics and data-governance requirements.

Dataset splits and preprocessing instructions for the public benchmarks will be released with the code.

📦 Pretrained Models

Pretrained checkpoints and configuration files will be provided after the public release.

Dataset	Configuration	Checkpoint
CAMELYON16	Coming soon	Coming soon
TCGA-Lung	Coming soon	Coming soon
TCGA-NSCLC	Coming soon	Coming soon
PANDA	Coming soon	Coming soon
✅ Reproducibility

To facilitate reproducibility, the public release will include:

dataset-specific configuration files;
five-fold split files for public datasets;
preprocessing and feature-loading instructions;
training and evaluation scripts;
pretrained checkpoints where redistribution is permitted;
pathway visualization utilities.

Because WSI experiments can be sensitive to feature extractors, data splits, and preprocessing settings, please ensure that the encoder, patch size, magnification, feature dimension, and split files match the target configuration.

📍 Acknowledgements

We thank the developers of the open-source computational pathology and multiple instance learning projects that support this work. Detailed acknowledgements and third-party code attributions will be added with the public code release.

Human brain tissue used in the study is provided by the National Human Brain Bank for Development and Function, Chinese Academy of Medical Sciences and Peking Union Medical College, with support from the relevant institutional brain banking and neuroscience resources. The use of the internal cohort follows the approved institutional ethics protocols described in the manuscript.

📌 Citation

If you find PathMIL useful in your research, please cite our paper:

@article{pathmil2026,
  title   = {PathMIL: Path-aware Multiple Instance Learning for Weakly Supervised Whole-Slide Image Classification},
  author  = {Author list to be updated},
  journal = {Manuscript under review},
  year    = {2026}
}

The final BibTeX entry will be updated after publication.

📄 License

Please refer to the LICENSE file for the terms of use. Dataset access and pretrained feature redistribution remain governed by the licenses of the original data providers and feature encoders.

📮 Contact

For questions, suggestions, or issues, please open a GitHub issue. Contact information for the corresponding authors will be added after the repository is finalized.
