# Paired GTA: Generating Driving Datasets for Controlled Photometric Shift Analysis

Additional material accompanying the manuscript submitted to **IEEE Transactions on Intelligent Vehicles**.

**Andrea Chianese, Giulio Rossolini, Marco Cococcioni, Alessandro Biondi, and Giorgio Buttazzo**

## Overview

Paired GTA is a GTA V-based framework for generating paired driving images under controlled illumination and weather changes. It reuses the scene configuration across **Day, Sunset, and Night**, combined with **Sunny, Rain, and Foggy** conditions.

This repository provides evaluation notebooks, plotting utilities, and additional quantitative and qualitative results for semantic segmentation under photometric shifts. The analysis measures prediction consistency with pseudo-labels obtained from the clean **Day–Sunny** reference image.

Please refer to the paper for the methodology, experimental protocol, and discussion.

## Repository contents

| Directory | Contents |
| --- | --- |
| [notebooks/](notebooks/) | Benchmark, plotting, ranking analysis, and generation-cost notebooks; model adapters and configuration files |
| [notebooks/segmentations/](notebooks/segmentations/) | Single-image segmentation notebooks and configurations |
| [out_cityscapes_final/](out_cityscapes_final/) | Evaluation CSVs, plots, and qualitative examples |
| [ranking_consistency_results/](ranking_consistency_results/) | Frame-wise prediction counts, correlation summaries, and ranking plots |
| [cityscapes_val_check/](cityscapes_val_check/) | Cityscapes validation summaries and per-class IoU results |

## Setup

Create a Python environment and install the main dependencies:

```bash
pip install torch torchvision transformers huggingface_hub
pip install jupyterlab numpy pandas matplotlib pillow pyyaml tqdm
```

Model inference additionally requires the image dataset and the checkpoints and external model implementations selected in the configuration. Adjust dataset, checkpoint, output, and import paths to your local setup before running the notebooks.

For plotting existing CSV results, only Jupyter, NumPy, pandas, and Matplotlib are needed.

## Main notebooks

- [Run the segmentation benchmark](notebooks/run_benchmarks_reference_metric.ipynb)
- [Plot benchmark results](notebooks/plot_benchmarks_notebook_updated_8models.ipynb)
- [Plot clean–adverse rankings](notebooks/plot_mirror_ranking_from_csv.ipynb)
- [Analyze generation costs](notebooks/generation_cost_analysis_updated.ipynb)
- [Extract single-image segmentations](notebooks/segmentations/)

The benchmark uses [this YAML configuration](notebooks/benchmark_config_reference_metric_mixed_loader_qualitative.yaml). Models can be selected there and extended through [the model registry](notebooks/benchmark_models_mixed_loader.py).

Supported models include **SegFormer-B0/B2/B5, Mask2Former Swin-S/Swin-L, DeepLabV3+ ResNet-101, PIDNet-M, and DDRNet-23**. SegFormer-B5 supplies the clean pseudo-label reference.

## Dataset organization

Place images in the illumination-specific directories:

```text
Dataset/
├── Day/
├── Sunset/
└── Night/
```

Use filenames of the form `<location_id>_<illumination>_<weather>.png`, with weather tokens `EXTRASUNNY`, `RAIN`, or `FOGGY`. Images sharing a location ID are grouped together; `Day_EXTRASUNNY` is the clean reference.

The notebooks export CSV summaries and PDF/PNG visualizations. Check their input and output paths relative to the notebook kernel's working directory before execution.
