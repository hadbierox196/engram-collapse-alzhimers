# Results

Committed output files from the simulation and ADNI validation pipeline. The raw sweep parquet is gitignored due to size — regenerate with `baseline_clean.py` if needed.

## week5_full_results.csv

Collapse thresholds and curve features per simulation configuration.  
540 rows (135 parameter combinations × 4 memory classes). All 540 entries collapsed.

Columns:

| Column | Description |
|---|---|
| N | Network size (500, 1000, 2000) |
| Overlap | Pattern overlap level (low, medium, high) |
| Deg_Rate | Degradation rate (slow, medium, fast) |
| Noise_Std | Retrieval noise σ (0.10–0.50) |
| Memory_Class | Memory class (weak\_remote, weak\_dense, strong\_recent, strong\_salient) |
| r\_star | Degradation ratio at first collapse below fidelity = 0.50 |
| fidelity\_at\_r | Mean fidelity at r\_star |
| fidelity\_prev | Mean fidelity at the step before r\_star |
| slope | (fidelity\_at\_r − fidelity\_prev) / deg\_step |
| collapsed | True for all 540 entries |
| braak\_at\_collapse | Braak stage label at r\_star |
| plateau\_height | Mean fidelity for r < (r\_star − 0.10) |
| post\_floor | Mean fidelity for r ≥ 0.90 |
| transition\_width | r-span from fidelity = 0.9 to fidelity = 0.1 |
| auc | Area under the full fidelity curve (trapezoid) |

## week6_failure_sequence.csv

Per-domain empirical decline slopes from the ADNI MCI→AD converter cohort (n = 191).

Columns:

| Column | Description |
|---|---|
| Domain | Cognitive domain (Episodic\_Memory, Working\_Memory, Semantic\_Memory) |
| N\_subjects | Number of subjects contributing valid observations |
| Mean\_slope | Mean OLS slope (z-score units per month) across subjects |
| CI\_lo / CI\_hi | 95% confidence interval on the mean slope |
| t\_stat | One-sample t-statistic (H₀: slope = 0) |
| p\_value | Two-tailed p-value |
| Empirical\_Rank | Rank by steepness of decline (1 = steepest) |

Model rank mapping used for Spearman correlation:  
Episodic = 1, Semantic = 2, Working = 3.  
Spearman ρ = 0.50, p = 0.667 (n = 3; not statistically significant).

## week5_sweep_results.parquet

Raw retrieval fidelity output (~8M rows). Not committed — gitignored.  
Regenerate with:

```python
from baseline_clean import *
cfg = SweepConfig()
df  = run_sensitivity_sweep(cfg)
df.to_parquet("results/week5_sweep_results.parquet", index=False)
```
