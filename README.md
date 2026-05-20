# Robust Semantic Segmentation under Adverse Conditions

This repository contains the official implementation of the benchmark introduced in our paper:

> **[Paper Title Placeholder]**  
> *Conference Title Submission*

---

## 🚀 Overview

We propose a benchmark to evaluate **robustness of semantic segmentation models under adverse conditions** (e.g., rain, fog, night) using paired images.

Unlike standard evaluation, we measure **prediction consistency** rather than absolute accuracy, focusing on how stable model predictions remain across environmental changes.

---

## 📊 Key Features

- Paired-image evaluation (clean vs adverse)
- Pixel-wise **masked agreement metric**
- Class-wise **retention analysis**
- Scenario-based evaluation (e.g., *Day-Rain*, *Sunset-Foggy*)
- Qualitative analysis of failure cases

---

## 📁 Repository Structure
	.
	├── notebooks/
	│   ├── plot_benchmarks_notebook.ipynb
	│   ├── plot_mirror_ranking_from_csv.ipynb
	│   ├── run_benchmarks_reference_metric.ipynb
	│   ├── extract_single_segmentation.ipynb
	│   ├── single_segmentation_config.yaml
	│   ├── benchmark_models.py
	│   └── benchmark_config_reference_metric...yaml
	│
	├── out_cityscapes_final/
	│   ├── *.csv
	│   ├── plots/
	│   ├── qualitative/
	│  
	├── ranking_consistency_results/
	│   ├── framewise_target_class_counts.csv
	│   ├── framewise_target_class_spearman.csv
	│   └── plots/
	│		└── mirror_ranking_car_4panels.pdf
	│		└── mirror_ranking_person_4panels.pdf
	│
	└── README.md

---

## ⚙️ Installation

Create a Python environment:

conda create -n segbench python=3.9  
conda activate segbench  

Install dependencies:

pip install torch torchvision  
pip install transformers huggingface_hub  
pip install matplotlib pandas tqdm scipy pyyaml  

---

## 📦 Dataset

Expected structure:

	Dataset/
	├── Day/
	├── Sunset/
	└── Night/

---

## ▶️ Running the Benchmark and Plots

Use:

	notebooks/run_benchmarks_reference_metric.ipynb
	notebooks/plot_benchmarks_notebook.ipynb
	notebooks/extract_single_segmentation.ipynb
	notebooks/plot_mirror_ranking_from_csv.ipynb

---

## 📈 Metrics

Masked Pixel Agreement:

Agreement = (1 / |A|) * Σ 1[p_clean == p_adv]

Class Retention:

Reference-based Retention_k = 

---

## 📊 Outputs

CSV:
- results_pixel_agreement_per_image.csv
- results_pixel_agreement_by_condition.csv
- results_class_retention_by_condition.csv
- framewise_target_class_counts.csv
- framewise_target_class_spearman.csv

Plots (PDF):
-	out_cityscapes_final/
	-	plots/	
   
-	ranking_consistency_results/
	-	plots/

---

## 🧠 Notes for Reviewers

- All plots are exported in **PDF format**
- No ground truth required
- Focus on prediction consistency
- In notebooks directory you can find "benchmark_models.py" where you can add/remove models to be used in the benchmark script

---

## 📜 License

Sant'Anna
