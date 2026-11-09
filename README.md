# Paired GTA: Generating Driving Datasets for Controlled Photometric Shift Analysis

**Andrea Chianese, Giulio Rossolini, Marco Cococcioni, Alessandro Biondi, and Giorgio Buttazzo**  
Companion repository for the manuscript submitted to *IEEE Transactions on Intelligent Vehicles*.

Paired GTA generates multiple views of a driving scene under different illumination and weather conditions. The pipeline uses a Python client and the VPilot, DeepGTAV, and ScriptHookV interfaces to GTA V. For each location it reuses the specified camera pose and scene configuration, including the selected dynamic objects and their placement, while changing time of day and weather. These pairs help isolate how segmentation predictions respond to photometric changes. Control is limited to the variables exposed through the game interfaces.

The study uses **101 locations**, with nine variants per location (Day, Sunset, Night × Sunny, Rain, Foggy) and zero to three inserted vehicles or pedestrians per scene. **Day–Sunny** (`Day_EXTRASUNNY` in filenames) is the clean reference.

## Start here

| What you want to inspect | Where to start |
| --- | --- |
| Class-wise reference consistency in Table I | [Archived class-wise scores](out_cityscapes_final/results_class_retention_by_condition.csv) and the [Table I recipe](#table-i-from-the-archived-csv) below |
| Per-image and condition-averaged agreement | [Per-image CSV](out_cityscapes_final/results_pixel_agreement_per_image.csv), [condition summary](out_cityscapes_final/results_pixel_agreement_by_condition.csv), and [run configuration](out_cityscapes_final/run_config_resolved.json) |
| Qualitative segmentation and failure cases | [SegFormer examples](out_cityscapes_final/qualitative_segformer/) and [DeepLabV3+ examples](out_cityscapes_final/qualitative_deeplabv3/) |
| Paired GTA vs. ACDC ranking analysis | [Frame-wise counts](ranking_consistency_results/framewise_target_class_counts.csv), [archived mirror plots](ranking_consistency_results/plots/), and the [ranking notes](#ranking-analysis) below |
| Generation costs | [100-scenario timing log](notebooks/100_execution_time.csv) and [timing notebook](notebooks/generation_cost_analysis_updated.ipynb) |
| Cityscapes validation diagnostics | [Validation summaries and per-class IoU CSVs](cityscapes_val_check/) |

This repository contains the **evaluation code, plotting notebooks, and archived outputs**. The raw Paired GTA and ACDC image sets, GTA V scene-generation code, and checkpoints loaded from local paths are separate inputs and are not included in this checkout. You can inspect the CSVs and rerun the plotting notebooks without GTA V, model weights, or a GPU.

## Evaluation protocol

SegFormer-B5 predicts the labels on each Day–Sunny image. The benchmark retains reference pixels with confidence **at least 0.60** and uses that same reference and mask for every evaluated model. It measures:

- **Masked pixel agreement:** the fraction of retained reference pixels where the prediction under the shifted condition equals the clean reference label. The condition summary averages these per-image scores.
- **Class-wise reference consistency (retention):** for each class and condition, the number of retained reference pixels of that class still assigned to it under the shift, divided by the total number of retained reference pixels of that class, pooled across pairs. The CSV exporter writes zero when there are no reference pixels for a class; check the reference pixel count when interpreting zeros.

The Cityscapes-trained models are SegFormer-B0/B2/B5, Mask2Former with Swin-S/Swin-L, DeepLabV3+ with ResNet-101, PIDNet-M, and DDRNet-23. SegFormer-B5 is included as the reference baseline and is excluded from best-model highlighting in Table I. Agreement with pseudo-labels measures **consistency with the clean reference**; it is not segmentation accuracy against human annotations. Errors in the reference can affect every model's scores.

### Table I from the archived CSV

Table I uses the class-wise CSV, with `Day-RAIN`, `Sunset-EXTRASUNNY`, and `Night-EXTRASUNNY` corresponding to Day–Rain, Sunset–Sunny, and Night–Sunny. Run the following from the repository root to reconstruct the values and the mean column:

```python
import pandas as pd

scores = pd.read_csv("out_cityscapes_final/results_class_retention_by_condition.csv")
classes = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "motorcycle", "bicycle",
]
for condition in ("Day-RAIN", "Sunset-EXTRASUNNY", "Night-EXTRASUNNY"):
    table = scores.loc[scores["sec_condition"].eq(condition)].pivot(
        index="model", columns="class_name", values="retention_masked"
    )[classes]
    table = (100 * table).round(1)
    table["Mean"] = table.mean(axis=1).round(1)
    print(f"\n{condition}\n{table.to_string()}")
```

### Ranking analysis

Section IV-C compares the ordering of frames by high-confidence pixel counts for `car` and `person` in Paired GTA and ACDC, using SegFormer-B5 and Mask2Former Swin-L. The paper counts pixels with confidence at least **0.90** for the predicted class. For each dataset, model, and class, it computes Spearman correlation between clean and adverse counts on the **25 frames with the largest clean counts**.

The checked-in [ranking notebook](notebooks/plot_mirror_ranking_from_csv.ipynb), [Spearman CSV](ranking_consistency_results/framewise_target_class_spearman.csv), and mirror PDFs report **all-frame** correlations. They illustrate the ranking workflow but do not reproduce Figure 6 or its exact scores. Applying the top-25 selection to the currently archived counts does not recover all the manuscript values either.

## Run the notebooks

For plots from existing CSVs, install `jupyterlab`, `numpy`, `pandas`, and `matplotlib` in a Python environment. Run Jupyter from the repository root. Check the **kernel's working directory** before using relative paths: the notebooks can open in their own directories.

| Notebook | Input and setup |
| --- | --- |
| [Benchmark plots](notebooks/plot_benchmarks_notebook_updated_8models.ipynb) | Set `RESULTS_DIR` to the actual path of [`out_cityscapes_final/`](out_cityscapes_final/). The notebook's default path points to a different folder. Its summary CSV is a masked-agreement table; use the recipe above for Table I's class-wise values. |
| [Mirror ranking plots](notebooks/plot_mirror_ranking_from_csv.ipynb) | Set `COUNTS_CSV` to the [archived counts](ranking_consistency_results/framewise_target_class_counts.csv) and `SAVE_DIR` to a separate output directory if you want to retain the archived PDFs. |
| [Generation cost analysis](notebooks/generation_cost_analysis_updated.ipynb) | Set `csv_path` to [`notebooks/100_execution_time.csv`](notebooks/100_execution_time.csv), or use the `notebooks/` working directory. If dependencies are installed, skip the `mamba install` cell. The notebook recovers the reported mean times (29.8 s pipeline, 4.0 s overhead, 33.8 s total per scenario) and produces diagnostic plots in a different layout from Figure 8. |
| [Run segmentation benchmark](notebooks/run_benchmarks_reference_metric.ipynb) | Supply the image dataset and model dependencies, then edit the [benchmark YAML](notebooks/benchmark_config_reference_metric_mixed_loader_qualitative.yaml). |
| [Single-image segmentation notebooks](notebooks/segmentations/) | Choose the Hugging Face, DeepLabV3+, PIDNet, or DDRNet notebook and its matching YAML. Make `notebooks/` available on the Python import path. |

### Rerun model inference

1. In addition to the plotting dependencies, install compatible versions of `torch` and `torchvision` for your device, plus `transformers`, `huggingface_hub`, `pillow`, `pyyaml`, and `tqdm`.
2. Use `notebooks/` as the kernel working directory for the benchmark, or adjust `CONFIG_PATH` and the Python import path. In the [YAML](notebooks/benchmark_config_reference_metric_mixed_loader_qualitative.yaml), set `data_dir`, `save_dir`, and `device` for your machine. The committed absolute paths are from the original setup.
3. Select the models under `models.enabled`. SegFormer and Mask2Former checkpoint IDs are defined in the [model registry](notebooks/benchmark_models_mixed_loader.py). DeepLabV3+, PIDNet, and DDRNet need compatible external implementations and checkpoints. Set their `checkpoint_path` values and adjust the notebook's external module paths, or disable those entries for a Hugging Face-only run. Keep `reference_model_key: segformer_b5` for the paper's protocol.
4. Keep the preprocessing selected by the [mixed dataloader](notebooks/dataloader_benchmark_mixed.py) and YAML: SegFormer and Mask2Former receive raw RGB tensors and use their own processors; the local adapters receive ImageNet-normalized tensors with `input_norm: none` to avoid double normalization.

The loader groups files by the location ID before the first underscore. Put each image in the directory for its illumination:

```text
Dataset/
├── Day/      <location_id>_Day_EXTRASUNNY.png, <location_id>_Day_RAIN.png, ...
├── Sunset/   <location_id>_Sunset_EXTRASUNNY.png, ...
└── Night/    <location_id>_Night_EXTRASUNNY.png, ...
```

Weather tokens are `EXTRASUNNY`, `RAIN`, and `FOGGY`. The default `dataloader_mode: random` selects one adverse view per clean image per pass. For a specific shift, use `dr` for Day–Rain, `ss` for Sunset–Sunny, or `ns` for Night–Sunny. Supply the requested variant for every location and check `sec_condition` in the output: the loader can fall back to another available image when a requested variant is missing. Fixed-condition runs need not reproduce the sample counts of the archived random run.

## Citation and third-party components

If you use this material, cite **Andrea Chianese, Giulio Rossolini, Marco Cococcioni, Alessandro Biondi, and Giorgio Buttazzo, “Paired GTA: Generating Driving Datasets for Controlled Photometric Shift Analysis,” manuscript submitted to IEEE Transactions on Intelligent Vehicles.**

GTA V, VPilot, DeepGTAV, ScriptHookV, and the evaluated models are third-party components. No separate repository license file is included in this checkout.
