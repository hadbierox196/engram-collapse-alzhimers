[![DOI](https://zenodo.org/badge/1264623579.svg)](https://doi.org/10.5281/zenodo.20727189)


# Engram-collapse-AD

Computational model of memory engram vulnerability in Alzheimer's disease.  
Attractor network predictions of sequential memory collapse across Braak stages, validated against ADNI longitudinal data.

---

## Overview

This repository contains the simulation code and analysis pipeline for a study of differential memory vulnerability in Alzheimer's disease. A Modern Hopfield Network (MHN) with four memory classes — defined by encoding strength (α), recency weight (ρ), sparsity, and salience (γ) — is subjected to Braak-stage-aware synaptic degradation. The model predicts a strict failure sequence: weak-remote memories collapse earliest, strong-salient memories last.

This predicted sequence is compared against empirically observed cognitive decline slopes in 191 MCI-to-AD converters from the ADNI cohort.

---

## Model

### Memory Classes

| Class | n | α | ρ | Sparsity | γ |
|---|---|---|---|---|---|
| strong\_salient | 10 | 2.0 | 1.0 | 0.05 | 1.0 |
| strong\_recent  | 10 | 1.8 | 1.0 | 0.05 | 0.0 |
| weak\_remote    | 15 | 0.8 | 0.2 | 0.10 | 0.0 |
| weak\_dense     | 15 | 0.6 | 0.5 | 0.20 | 0.0 |

### Braak Stage Degradation Protocol

| Stage | r range | Mechanism |
|---|---|---|
| 0 | 0.00 | Healthy baseline |
| 1 (Transentorhinal) | 0.00–0.33 | Class-differentiated synaptic noise; weak\_remote most vulnerable |
| 2 (Limbic) | 0.33–0.66 | Salience collapse (γ decay) + secondary attenuation |
| 3 (Isocortical) | 0.66–1.00 | Global multiplicative collapse + β floor |

The MHN uses decoupled competition and reconstruction matrices. `Xi_query` (unit-norm) governs softmax pattern selection; `Xi_recon` (α·ρ scaled) governs output amplitude. This preserves the encoding-strength hierarchy without suppressing weak-class attractors during competition.

---

## Results

### Simulation — 540 configurations (N × overlap × noise × degradation rate)

Collapse threshold r\* is defined as the degradation ratio at which mean cosine fidelity first drops below 0.50.

| Memory Class | Mean r\* | AUC (N=500→2000) |
|---|---|---|
| weak\_remote    | 0.756 | 0.694 → 0.778 (+12.0%) |
| weak\_dense     | 0.883 | 0.836 → 0.873 (+4.4%)  |
| strong\_recent  | 0.964 | 0.938 → 0.944 (+0.7%)  |
| strong\_salient | 0.983 | 0.945 → 0.964 (+2.1%)  |

Failure sequence is strictly ordered and robust across all 540 configurations.  
Main effect of memory class: F(3,536) = 13,187.93, p < 2.2×10⁻¹⁶, η² = 0.987.

### ADNI Empirical Validation — 191 MCI→AD converters

Per-domain cognitive decline slopes (OLS on z-scores, ±36-month conversion window):

| Domain | Slope (z/month) | 95% CI | p | Empirical rank | Model rank |
|---|---|---|---|---|---|
| Episodic Memory (RAVLT) | −0.02487 | [−0.034, −0.016] | < 0.001 | 1 | 1 ✓ |
| Working Memory (Digit Span) | −0.02420 | [−0.034, −0.015] | < 0.001 | 2 | 3 ✗ |
| Semantic Memory (BNT) | −0.02184 | [−0.030, −0.014] | < 0.001 | 3 | 2 ✗ |

Episodic memory rank matches the model prediction. Working and semantic memory ranks are transposed relative to the model.  
Spearman rank correlation: ρ = 0.50, p = 0.667 (n = 3 domains). Not statistically significant — at n = 3, a perfect rank ordering is required to achieve p < 0.05 (p_min = 0.167), making this test uninformative either way.

---

## Repository Structure

```
engram-collapse-AD/
│
├── baseline_clean.py       # MHN simulation + full analysis pipeline
├── week7_figures.py        # Publication figure generation (Figures 1–4)
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── results/
│   ├── week5_full_results.csv        # Collapse thresholds + curve features (540 configs)
│   ├── week6_failure_sequence.csv    # ADNI per-domain decline slopes + rank alignment
│   └── README.md
│
├── figures/                # PNG + SVG outputs from week7_figures.py
│   ├── fig1_fidelity_curves.png / .svg
│   ├── fig2_sensitivity_panel.png / .svg
│   ├── fig3_slopegraph_adni.png / .svg
│   └── fig4_auc_by_N.png / .svg
│
└── data/
    └── README.md           # ADNI data access instructions (data not included)
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/engram-collapse-AD.git
cd engram-collapse-AD
pip install -r requirements.txt

# Smoke test — single config, ~30 seconds
python baseline_clean.py

# Generate all four publication figures from committed results
python week7_figures.py

# Full sweep — 135 configs × degradation steps, ~8–12 min on GPU
# Uncomment cfg_full block in baseline_clean.py __main__
```

---

## Reproducing the Full Pipeline

```python
# 1. Run sweep
from baseline_clean import *
cfg_full  = SweepConfig()
df_sweep  = run_sensitivity_sweep(cfg_full)
df_sweep.to_parquet("results/week5_sweep_results.parquet", index=False)

# 2. Extract features
df_thresh   = extract_collapse_thresholds(df_sweep)
df_features = compute_curve_features(df_sweep, df_thresh)
df_features.to_csv("results/week5_full_results.csv", index=False)

# 3. ADNI validation (requires ADNI data — see data/README.md)
cohort = build_adni_cohort("data/")
df_seq = compute_adni_failure_sequence(cohort)
df_seq.to_csv("results/week6_failure_sequence.csv", index=False)

# 4. Figures
# python week7_figures.py
```

---

## ADNI Data

Validation uses longitudinal data from the [Alzheimer's Disease Neuroimaging Initiative](https://adni.loni.usc.edu/). ADNI data cannot be redistributed. To reproduce the validation analysis, register at adni.loni.usc.edu, download `DXSUM_*.csv` and `NEUROBAT_*.csv`, and place them in `data/`. See `data/README.md` for full instructions.

---

## Dependencies

```
torch >= 2.1.0
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
scipy >= 1.11.0
seaborn >= 0.12.0
pingouin >= 0.5.4
pyarrow >= 14.0.0
```

---

## Citation

```(https://doi.org/10.5281/zenodo.20727189
```

---

## License

MIT
