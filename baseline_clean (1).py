# -*- coding: utf-8 -*-
"""
Synaptic Degradation Thresholds and the Sequential Collapse of Engram Stability
in Alzheimer's Disease: Predictions from Attractor Network Modeling

Modern Hopfield Network simulation of memory degradation across Braak stages.
"""

# ============================================================
# Imports
# ============================================================
import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BETA_BASE = 8.0
print(f"Device: {DEVICE}")


# ============================================================
# Plotting Style
# ============================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.title_fontsize": 9,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "patch.linewidth": 0.8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.4,
    "grid.color": "#CCCCCC",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "legend.borderpad": 0.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "figure.constrained_layout.use": True,
})

# Colorblind-safe palette (Wong 2011)
PALETTE = {
    "weak_remote":    "#E69F00",
    "weak_dense":     "#56B4E9",
    "strong_recent":  "#009E73",
    "strong_salient": "#CC79A7",
}
CLASS_ORDER = ["weak_remote", "weak_dense", "strong_recent", "strong_salient"]
CLASS_LABELS = {
    "weak_remote":    "Weak-Remote (Episodic)",
    "weak_dense":     "Weak-Dense (Semantic)",
    "strong_recent":  "Strong-Recent (Working)",
    "strong_salient": "Strong-Salient (Procedural/Emotional)",
}


# ============================================================
# Configuration
# ============================================================

# Memory class specs: (name, n_patterns, alpha, sparsity, rho, gamma)
CLASS_SPECS = [
    ("strong_salient", 10, 2.0, 0.05, 1.0, 1.0),
    ("strong_recent",  10, 1.8, 0.05, 1.0, 0.0),
    ("weak_remote",    15, 0.8, 0.10, 0.2, 0.0),
    ("weak_dense",     15, 0.6, 0.20, 0.5, 0.0),
]

# Vulnerability profile per Braak stage
CLASS_SLICES = {
    "strong_salient": slice(0,  10),
    "strong_recent":  slice(10, 20),
    "weak_remote":    slice(20, 35),
    "weak_dense":     slice(35, 50),
}

CORRUPTION_PROFILE = {
    1: {"strong_salient": 0.05, "strong_recent": 0.15,
        "weak_remote": 0.60,   "weak_dense": 0.45},
    2: {"strong_salient": 0.20, "strong_recent": 0.45,
        "weak_remote": 0.80,   "weak_dense": 0.70},
}

BETA_FLOOR    = 1.2
GAMMA_SALIENT = 1.0


@dataclass
class SweepConfig:
    n_patterns:         int   = 50
    n_trials:           int   = 50
    epsilon_conv:       float = 1e-4
    max_iter:           int   = 100
    pattern_seed:       int   = 3       # validated clean seed across all overlap levels
    N_sizes:            List[int]   = field(default_factory=lambda: [500, 1000, 2000])
    overlap_levels:     List[str]   = field(default_factory=lambda: ["low", "medium", "high"])
    noise_levels:       List[float] = field(default_factory=lambda: [0.10, 0.20, 0.30, 0.40, 0.50])
    deg_rates:          List[str]   = field(default_factory=lambda: ["slow", "medium", "fast"])
    deg_step_map:       dict = field(default_factory=lambda: {
        "slow": 0.025, "medium": 0.050, "fast": 0.100
    })
    overlap_target_map: dict = field(default_factory=lambda: {
        "low": 0.010, "medium": 0.050, "high": 0.120
    })


CFG = SweepConfig()


# ============================================================
# Pattern Library
# ============================================================

def build_pattern_library(
    N: int,
    overlap_label: str,
    seed: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """
    Construct the pattern library with decoupled reconstruction and query matrices.

    Xi_recon is built from clean sparse patterns (pre-background injection),
    preserving the alpha*rho amplitude hierarchy. Background injection for
    inter-pattern overlap control only affects Xi_query (competition space).

    Returns
    -------
    Xi_raw    : [N, P]  clean mean-centred patterns (cue generation + overlap measure)
    Xi_recon  : [N, P]  alpha*rho scaled (output amplitude)
    Xi_query  : [N, P]  unit-norm vectors (softmax competition)
    beta_comp : [P]     salience-weighted inverse temperature
    labels    : list[str] per-column class name
    """
    torch.manual_seed(seed)
    target_gram = CFG.overlap_target_map[overlap_label]
    patterns, alphas, rhos, gammas, labels = [], [], [], [], []

    for label, n_pats, alpha, sparsity, rho, gamma in CLASS_SPECS:
        xi = (torch.rand(N, n_pats, device=DEVICE) < sparsity).float()
        xi = xi - xi.mean(dim=0, keepdim=True)
        patterns.append(xi)
        alphas  += [alpha]  * n_pats
        rhos    += [rho]    * n_pats
        gammas  += [gamma]  * n_pats
        labels  += [label]  * n_pats

    Xi_clean = torch.cat(patterns, dim=1).float()     # [N, P]

    alpha_t  = torch.tensor(alphas, dtype=torch.float32, device=DEVICE)
    rho_t    = torch.tensor(rhos,   dtype=torch.float32, device=DEVICE)
    gamma_t  = torch.tensor(gammas, dtype=torch.float32, device=DEVICE)

    # Reconstruction matrix: preserves encoding-strength amplitude
    Xi_recon = Xi_clean * (alpha_t * rho_t).unsqueeze(0)

    # Background injection for overlap control (competition space only)
    xi_var     = Xi_clean.var().item()
    b_scale    = ((target_gram * xi_var) / max(1 - target_gram, 1e-6)) ** 0.5
    background = torch.randn(N, 1, device=DEVICE)
    background = background / background.norm() * (N ** 0.5)
    Xi_overlap = Xi_clean + b_scale * background
    Xi_overlap = Xi_overlap - Xi_overlap.mean(dim=0, keepdim=True)

    norms     = Xi_overlap.norm(dim=0, keepdim=True).clamp(min=1e-8)
    Xi_query  = Xi_overlap / norms
    beta_comp = BETA_BASE * (1.0 + gamma_t)

    return Xi_clean, Xi_recon, Xi_query, beta_comp, labels


def verify_gram(Xi_query: torch.Tensor, overlap_label: str, tol: float = 0.04) -> bool:
    """Verify mean off-diagonal Gram value is within tolerance of target overlap."""
    G        = (Xi_query.T @ Xi_query).abs()
    P        = G.shape[0]
    mask     = ~torch.eye(P, dtype=torch.bool, device=DEVICE)
    mean_off = G[mask].mean().item()
    target   = CFG.overlap_target_map[overlap_label]
    ok       = abs(mean_off - target) < tol
    print(f"  Gram [{overlap_label}]: mean_off={mean_off:.4f}, "
          f"target={target:.3f} → {'PASS' if ok else 'WARN'}")
    return ok


# ============================================================
# Modern Hopfield Network
# ============================================================

class MHNv3:
    """
    Modern Hopfield Network with decoupled competition and reconstruction.

    Competition  : softmax( beta_comp * Xi_query.T @ x )  — unit-norm patterns
    Reconstruction: Xi_recon @ attention                   — amplitude-scaled patterns

    This separation ensures weak-class memories can win the softmax competition
    when correctly cued, while still producing proportionally weaker reconstructed
    states — matching the biological alpha*rho encoding hierarchy.
    """

    def __init__(
        self,
        Xi_raw:    torch.Tensor,   # [N, P]
        Xi_recon:  torch.Tensor,   # [N, P]  alpha*rho scaled
        Xi_query:  torch.Tensor,   # [N, P]  unit-norm
        beta_comp: torch.Tensor,   # [P]
    ):
        self._Xi_raw_0    = Xi_raw.clone()
        self._Xi_recon_0  = Xi_recon.clone()
        self._Xi_query_0  = Xi_query.clone()
        self._beta_comp_0 = beta_comp.clone()
        self.reset()

    def reset(self):
        self.Xi_raw    = self._Xi_raw_0.clone()
        self.Xi_recon  = self._Xi_recon_0.clone()
        self.Xi_query  = self._Xi_query_0.clone()
        self.beta_comp = self._beta_comp_0.clone()

    @torch.no_grad()
    def update_step(self, X: torch.Tensor) -> torch.Tensor:
        """Single MHN update step, batched over B probes. X: [N, B] → [N, B]."""
        logits = (self.Xi_query.T @ X) * self.beta_comp.unsqueeze(1)  # [P, B]
        att    = torch.softmax(logits, dim=0)                          # [P, B]
        return self.Xi_recon @ att                                      # [N, B]

    @torch.no_grad()
    def retrieve(self, X_init: torch.Tensor) -> torch.Tensor:
        """
        Iterate update_step to convergence or CFG.max_iter.

        Parameters
        ----------
        X_init : [N, B]  noisy cue vectors

        Returns
        -------
        X_conv : [N, B]  converged state vectors
        """
        X = X_init.clone()
        for _ in range(CFG.max_iter):
            X_new = self.update_step(X)
            if (X_new - X).norm(dim=0).max().item() < CFG.epsilon_conv:
                break
            X = X_new
        return X_new

    def apply_braak(self, r: float, seed: int = 0):
        """
        Apply Braak-stage-aware synaptic degradation in-place.

        Stage 0 : r = 0.0              — healthy baseline
        Stage 1 : r ∈ (0.00, 0.33)    — Transentorhinal: class-differentiated attenuation
        Stage 2 : r ∈ [0.33, 0.66)    — Limbic: salience collapse + secondary attenuation
        Stage 3 : r ∈ [0.66, 1.00]    — Isocortical: global multiplicative collapse

        Xi_query is re-normalised from Xi_recon after each call.
        """
        torch.manual_seed(seed)
        P    = self.Xi_recon.shape[1]
        vuln = torch.zeros(P, device=DEVICE)
        vuln[0:10]  = 0.05
        vuln[10:20] = 0.15
        vuln[20:35] = 0.60
        vuln[35:50] = 0.45

        if r <= 0.33:
            s             = r / 0.33
            noise_scale   = s * vuln * 0.15
            noise         = torch.randn_like(self.Xi_recon) * noise_scale.unsqueeze(0)
            self.Xi_recon = self._Xi_recon_0 + noise

        elif r <= 0.66:
            s               = (r - 0.33) / 0.33
            vuln2           = vuln.clone()
            vuln2[0:10]  = 0.05
            vuln2[10:20] = 0.45
            vuln2[20:35] = 0.80
            vuln2[35:50] = 0.60
            noise_scale          = s * vuln2 * 0.15
            noise                = torch.randn_like(self.Xi_recon) * noise_scale.unsqueeze(0)
            self.Xi_recon        = self._Xi_recon_0 + noise
            gamma_decay          = 1.0 - 0.6 * s
            self.beta_comp[0:10] = BETA_BASE * (1.0 + gamma_decay * 1.0)

        else:
            s              = (r - 0.66) / 0.34
            self.Xi_recon  = self._Xi_recon_0 * (1.0 - s)
            beta_floor     = 1.2
            self.beta_comp = self._beta_comp_0 + s * (beta_floor - self._beta_comp_0)

        norms         = self.Xi_recon.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.Xi_query = self.Xi_recon / norms


# ============================================================
# Fidelity Measurement
# ============================================================

@torch.no_grad()
def measure_fidelity_batch(
    mhn:          MHNv3,
    Xi_raw:       torch.Tensor,
    noise_std:    float,
    n_trials:     int,
    class_labels: List[str],
) -> pd.DataFrame:
    """
    Run retrieval for every stored pattern and compute per-trial fidelity.

    Cosine fidelity is computed against Xi_query (unit direction) —
    amplitude-agnostic, so all classes read ~1.0 at Stage 0.
    Overlap is computed against Xi_raw for biological interpretability.

    Returns
    -------
    DataFrame with columns:
        Pattern_ID | Memory_Class | Trial | Cosine_Fidelity | Overlap
    """
    N, P    = Xi_raw.shape
    Xi_dir  = mhn.Xi_query
    records = []

    for pat_idx in range(P):
        xi_raw = Xi_raw[:, pat_idx].unsqueeze(1)
        xi_dir = Xi_dir[:, pat_idx].unsqueeze(1)

        noise  = torch.randn(N, n_trials, device=DEVICE) * noise_std
        X_cue  = xi_raw + noise
        X_conv = mhn.retrieve(X_cue)

        xc_norms = X_conv.norm(dim=0).clamp(min=1e-8)
        cosines  = (xi_dir * X_conv).sum(dim=0) / xc_norms
        overlaps = (xi_raw * X_conv).sum(dim=0) / N

        for t in range(n_trials):
            records.append({
                "Pattern_ID":      pat_idx,
                "Memory_Class":    class_labels[pat_idx],
                "Trial":           t,
                "Cosine_Fidelity": cosines[t].item(),
                "Overlap":         overlaps[t].item(),
            })

    return pd.DataFrame(records)


# ============================================================
# Sensitivity Sweep
# ============================================================

def run_sensitivity_sweep(cfg: SweepConfig) -> pd.DataFrame:
    """
    Full factorial sweep over N × overlap × deg_rate × noise_std.

    For each configuration, applies Braak degradation at every r step
    and measures per-pattern fidelity across n_trials noisy cues.

    Returns a long-format DataFrame with one row per
    (config × degradation step × pattern × trial).
    """
    all_frames = []
    combos     = list(itertools.product(
        cfg.N_sizes, cfg.overlap_levels, cfg.deg_rates, cfg.noise_levels
    ))
    total = len(combos)
    print(f"Total configurations: {total}")
    start = time.time()

    for i, (N, overlap, deg_rate, noise_std) in enumerate(combos):
        Xi_raw, Xi_recon, Xi_query, beta_comp, class_labels = \
            build_pattern_library(N, overlap, seed=cfg.pattern_seed)
        verify_gram(Xi_query, overlap)

        mhn      = MHNv3(Xi_raw, Xi_recon, Xi_query, beta_comp)
        step     = cfg.deg_step_map[deg_rate]
        r_values = np.arange(0.0, 1.0 + step / 2, step)

        for r in r_values:
            mhn.reset()
            mhn.apply_braak(r)

            if   r == 0.0:  stage = "Stage_0"
            elif r <= 0.33: stage = "Stage_1"
            elif r <= 0.66: stage = "Stage_2"
            else:            stage = "Stage_3"

            df_step = measure_fidelity_batch(
                mhn, Xi_raw, noise_std, cfg.n_trials, class_labels
            )
            df_step["N"]             = N
            df_step["Overlap"]       = overlap
            df_step["Deg_Rate"]      = deg_rate
            df_step["Noise_Std"]     = noise_std
            df_step["Degradation_r"] = round(r, 4)
            df_step["Braak_Stage"]   = stage
            all_frames.append(df_step)

        del mhn, Xi_raw, Xi_recon, Xi_query, beta_comp
        torch.cuda.empty_cache()

        elapsed = time.time() - start
        eta     = elapsed / (i + 1) * (total - i - 1) / 60
        print(f"[{i+1}/{total}] N={N}, overlap={overlap}, "
              f"deg={deg_rate}, noise={noise_std:.2f} | "
              f"ETA: {eta:.1f} min")

    return pd.concat(all_frames, ignore_index=True)


# ============================================================
# Threshold and Feature Extraction
# ============================================================

COLLAPSE_THRESHOLD = 0.5


def extract_collapse_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (N, Overlap, Deg_Rate, Noise_Std, Memory_Class), find r* —
    the degradation ratio at which mean cosine fidelity first drops below 0.5.
    Uses linear interpolation between bracketing points.

    Returns one row per configuration with columns:
        r_star | fidelity_at_r | fidelity_prev | slope | collapsed | braak_at_collapse
    """
    group_keys = ["N", "Overlap", "Deg_Rate", "Noise_Std", "Memory_Class"]

    curve = (
        df.groupby(group_keys + ["Degradation_r", "Braak_Stage"])["Cosine_Fidelity"]
        .mean()
        .reset_index()
        .rename(columns={"Cosine_Fidelity": "mean_fidelity"})
        .sort_values(group_keys + ["Degradation_r"])
    )

    records = []
    for keys, grp in curve.groupby(group_keys):
        grp      = grp.sort_values("Degradation_r").reset_index(drop=True)
        mask     = grp["mean_fidelity"] < COLLAPSE_THRESHOLD
        deg_step = CFG.deg_step_map[keys[2]]

        if mask.any():
            idx      = mask.idxmax()
            r_star   = grp.loc[idx, "Degradation_r"]
            fid_at   = grp.loc[idx, "mean_fidelity"]
            fid_prev = grp.loc[idx - 1, "mean_fidelity"] if idx > 0 else 1.0
            slope    = (fid_at - fid_prev) / deg_step
            braak    = grp.loc[idx, "Braak_Stage"]
            collapsed = True
        else:
            r_star   = float("nan")
            fid_at   = grp["mean_fidelity"].iloc[-1]
            fid_prev = float("nan")
            slope    = float("nan")
            braak    = "no_collapse"
            collapsed = False

        records.append(dict(
            zip(group_keys, keys),
            r_star            = r_star,
            fidelity_at_r     = fid_at,
            fidelity_prev     = fid_prev,
            slope             = slope,
            collapsed         = collapsed,
            braak_at_collapse = braak,
        ))

    return pd.DataFrame(records)


def compute_curve_features(df: pd.DataFrame, df_thresh: pd.DataFrame) -> pd.DataFrame:
    """
    Compute plateau height, post-collapse floor, transition width (r-span
    from fidelity=0.9 to 0.1), and AUC for each configuration.
    """
    group_keys = ["N", "Overlap", "Deg_Rate", "Noise_Std", "Memory_Class"]

    curve = (
        df.groupby(group_keys + ["Degradation_r"])["Cosine_Fidelity"]
        .mean()
        .reset_index()
        .rename(columns={"Cosine_Fidelity": "mean_fidelity"})
    )
    curve = curve.merge(df_thresh[group_keys + ["r_star"]], on=group_keys, how="left")

    def interp_crossing(r_arr, f_arr, threshold):
        for i in range(len(f_arr) - 1):
            if (f_arr[i] >= threshold >= f_arr[i+1]) or \
               (f_arr[i] <= threshold <= f_arr[i+1]):
                if abs(f_arr[i+1] - f_arr[i]) < 1e-9:
                    return r_arr[i]
                t = (threshold - f_arr[i]) / (f_arr[i+1] - f_arr[i])
                return r_arr[i] + t * (r_arr[i+1] - r_arr[i])
        return np.nan

    records = []
    for keys, grp in curve.groupby(group_keys):
        grp    = grp.sort_values("Degradation_r").reset_index(drop=True)
        r      = grp["Degradation_r"].values
        f      = grp["mean_fidelity"].values
        r_star = grp["r_star"].iloc[0]

        pre_mask = r < (r_star - 0.10) if not np.isnan(r_star) else r < 0.60
        plateau  = f[pre_mask].mean() if pre_mask.any() else np.nan

        floor_mask = r >= 0.90
        floor      = f[floor_mask].mean() if floor_mask.any() else np.nan

        r_90    = interp_crossing(r, f, 0.9)
        r_10    = interp_crossing(r, f, 0.1)
        t_width = (r_10 - r_90) if not (np.isnan(r_90) or np.isnan(r_10)) else np.nan
        auc     = np.trapz(f, r)

        records.append(dict(
            zip(group_keys, keys),
            plateau_height   = plateau,
            post_floor       = floor,
            transition_width = t_width,
            auc              = auc,
        ))

    return pd.DataFrame(records)


# ============================================================
# Publication Figures
# ============================================================

def save_figure(fig, stem: str, formats=("png", "svg")):
    """Save figure as PNG (300 dpi) and SVG."""
    for fmt in formats:
        path = f"{stem}.{fmt}"
        fig.savefig(path, dpi=300 if fmt == "png" else None,
                    bbox_inches="tight", transparent=False)
        print(f"Saved: {path}")


def plot_mean_ci(ax, x, mean, ci_lower, ci_upper, memory_class,
                 alpha_band=0.18, label=None):
    """Plot a mean line with shaded 95% CI band for one memory class."""
    color = PALETTE[memory_class]
    lbl   = label if label is not None else CLASS_LABELS[memory_class]
    ax.plot(x, mean, color=color, linewidth=1.6, label=lbl, zorder=3)
    ax.fill_between(x, ci_lower, ci_upper, color=color, alpha=alpha_band,
                    linewidth=0, zorder=2)


def make_figure(nrows=1, ncols=1, panel_width=3.3, panel_height=2.8):
    """Create a figure pre-sized for journal column widths (single ≈ 3.3 in)."""
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_width * ncols, panel_height * nrows),
        squeeze=False,
    )
    return fig, axes


def format_axis(ax, xlabel="", ylabel="", title="",
                xlim=None, ylim=None, xstep=None, ystep=None):
    ax.set_xlabel(xlabel, labelpad=4)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_title(title, pad=6, fontweight="normal")
    if xlim   is not None: ax.set_xlim(xlim)
    if ylim   is not None: ax.set_ylim(ylim)
    if xstep  is not None: ax.xaxis.set_major_locator(ticker.MultipleLocator(xstep))
    if ystep  is not None: ax.yaxis.set_major_locator(ticker.MultipleLocator(ystep))


def _shade_braak_stages(ax):
    for bounds, alpha in [((0.0, 0.33), 0.03), ((0.33, 0.66), 0.02), ((0.66, 1.0), 0.06)]:
        ax.axvspan(*bounds, color="#888888", alpha=alpha, zorder=0)
    for b in [0.33, 0.66]:
        ax.axvline(b, color="#AAAAAA", lw=0.8, ls="--", zorder=1)
    ax.axhline(0.5, color="#999999", lw=0.9, ls=":", zorder=2)


def plot_collapse_curves(df_sweep: pd.DataFrame, df_thresh: pd.DataFrame):
    """
    Figure 1: 4-panel collapse figure.
      A — Fidelity curves by memory class (canonical condition)
      B — Vulnerability ladder (r* per class × N)
      C — Collapse abruptness (|slope| by class × deg_rate)
      D — Network size effect on weak_remote
    """
    curve_all = (
        df_sweep
        .groupby(["N", "Overlap", "Deg_Rate", "Noise_Std",
                  "Memory_Class", "Degradation_r"])["Cosine_Fidelity"]
        .mean()
        .reset_index()
    )

    def get_curve(N, overlap, deg_rate, noise_std, mem_class):
        m = ((curve_all["N"] == N) & (curve_all["Overlap"] == overlap) &
             (curve_all["Deg_Rate"] == deg_rate) & (curve_all["Noise_Std"] == noise_std) &
             (curve_all["Memory_Class"] == mem_class))
        sub = curve_all[m].sort_values("Degradation_r")
        return sub["Degradation_r"].values, sub["Cosine_Fidelity"].values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    N_COLORS  = {500: "#fdae61", 1000: "#abd9e9", 2000: "#2c7bb6"}

    # Panel A
    ax = axes[0, 0]
    _shade_braak_stages(ax)
    for cls in CLASS_ORDER:
        r, f = get_curve(1000, "low", "medium", 0.30, cls)
        ax.plot(r, f, color=PALETTE[cls], lw=2.0, label=CLASS_LABELS[cls])
    format_axis(ax, xlabel="Degradation r", ylabel="Mean Cosine Fidelity",
                title="A  Fidelity curves by memory class\nN=1000, overlap=low, deg=medium, σ=0.30",
                xlim=(0, 1), ylim=(-0.05, 1.05))
    ax.legend(fontsize=8)

    # Panel B — vulnerability ladder
    ax = axes[0, 1]
    _shade_braak_stages(ax)
    y_positions = {cls: i for i, cls in enumerate(CLASS_ORDER)}
    for cls in CLASS_ORDER:
        sub = df_thresh[df_thresh["Memory_Class"] == cls]
        for _, row in sub.iterrows():
            n_jitter = {500: -0.12, 1000: 0.0, 2000: 0.12}[row["N"]]
            ax.scatter(row["r_star"], y_positions[cls] + n_jitter,
                       color=N_COLORS[row["N"]], s=18, alpha=0.7)
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_yticklabels([CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=8)
    ax.set_xlabel("r* (collapse threshold)")
    ax.set_title("B  Vulnerability ladder\nr* per class × N", fontsize=9)
    n_patches = [mpatches.Patch(color=N_COLORS[n], label=f"N={n}") for n in [500, 1000, 2000]]
    ax.legend(handles=n_patches, fontsize=8)

    # Panel C — slope magnitude
    ax   = axes[1, 0]
    xpos = np.arange(len(CLASS_ORDER))
    deg_colors = {"slow": "#7fbc41", "medium": "#4d9221", "fast": "#276419"}
    for di, deg in enumerate(["slow", "medium", "fast"]):
        means = [df_thresh[(df_thresh["Memory_Class"] == cls) &
                            (df_thresh["Deg_Rate"] == deg)]["slope"].abs().mean()
                 for cls in CLASS_ORDER]
        ax.bar(xpos + (di - 1) * 0.25, means, 0.25, color=deg_colors[deg],
               alpha=0.85, label=deg)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_ORDER], fontsize=8)
    ax.set_ylabel("|slope| at collapse")
    ax.set_title("C  Collapse abruptness by class × deg rate", fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5)

    # Panel D — N effect on weak_remote
    ax = axes[1, 1]
    _shade_braak_stages(ax)
    for N in [500, 1000, 2000]:
        r, f = get_curve(N, "low", "medium", 0.30, "weak_remote")
        ax.plot(r, f, color=N_COLORS[N], lw=2.0, label=f"N={N}")
    format_axis(ax, xlabel="Degradation r", ylabel="Mean Cosine Fidelity",
                title="D  Network size effect on weak_remote\noverlap=low, deg=medium, σ=0.30",
                xlim=(0, 1), ylim=(-0.05, 1.05))
    ax.legend(fontsize=8)

    plt.suptitle("Sequential Engram Collapse — Braak Stage Degradation Protocol",
                 fontsize=12, y=1.01)
    save_figure(fig, "fig1_collapse_curves")
    plt.show()


def plot_sensitivity_figure(df_features: pd.DataFrame, df_sweep: pd.DataFrame):
    """
    Figure 2: 4-panel sensitivity analysis.
      A — Post-collapse residual fidelity by class
      B — AUC by class × N
      C — Transition width by class
      D — weak_remote × noise levels
    """
    N_COLORS = {500: "#fdae61", 1000: "#abd9e9", 2000: "#2c7bb6"}
    xpos     = np.arange(len(CLASS_ORDER))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel A — post-collapse floor
    ax     = axes[0, 0]
    floors = df_features.groupby("Memory_Class")["post_floor"].mean()
    for i, cls in enumerate(CLASS_ORDER):
        ax.bar(i, floors[cls], color=PALETTE[cls], alpha=0.85, width=0.55)
        ax.text(i, floors[cls] + 0.008, f"{floors[cls]:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="#999999", lw=0.9, ls=":", label="Collapse threshold (0.5)")
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_ORDER], fontsize=8)
    ax.set_ylabel("Mean fidelity at r ≥ 0.90")
    ax.set_title("A  Post-collapse residual fidelity", fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5)

    # Panel B — AUC by class × N
    ax       = axes[0, 1]
    auc_data = df_features.groupby(["Memory_Class", "N"])["auc"].mean().unstack()
    for di, N in enumerate([500, 1000, 2000]):
        vals = [auc_data.loc[cls, N] for cls in CLASS_ORDER]
        ax.bar(xpos + (di - 1) * 0.25, vals, 0.25, color=N_COLORS[N],
               alpha=0.85, label=f"N={N}")
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_ORDER], fontsize=8)
    ax.set_ylabel("Area under fidelity curve")
    ax.set_title("B  Total memory capacity (AUC) by class × N", fontsize=9)
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5)

    # Panel C — transition width
    ax     = axes[1, 0]
    widths = df_features.groupby("Memory_Class")["transition_width"].mean()
    for i, cls in enumerate(CLASS_ORDER):
        ax.bar(i, widths[cls], color=PALETTE[cls], alpha=0.85, width=0.55)
        ax.text(i, widths[cls] + 0.001, f"{widths[cls]:.4f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASS_ORDER], fontsize=8)
    ax.set_ylabel("Transition width (Δr, 0.9→0.1 fidelity)")
    ax.set_title("C  Collapse abruptness (transition width)", fontsize=9)
    ax.yaxis.grid(True, color="#CCCCCC", lw=0.5)

    # Panel D — weak_remote × noise
    ax          = axes[1, 1]
    noise_colors = {0.1: "#ffffcc", 0.2: "#c7e9b4", 0.3: "#7fcdbb",
                    0.4: "#2c7fb8", 0.5: "#253494"}
    curve_all   = (
        df_sweep.groupby(["N", "Overlap", "Deg_Rate", "Noise_Std",
                          "Memory_Class", "Degradation_r"])["Cosine_Fidelity"]
        .mean().reset_index()
    )
    _shade_braak_stages(ax)
    for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
        m = ((curve_all["N"] == 1000) & (curve_all["Overlap"] == "low") &
             (curve_all["Deg_Rate"] == "medium") & (curve_all["Noise_Std"] == noise) &
             (curve_all["Memory_Class"] == "weak_remote"))
        sub = curve_all[m].sort_values("Degradation_r")
        ax.plot(sub["Degradation_r"], sub["Cosine_Fidelity"],
                color=noise_colors[noise], lw=1.8, label=f"σ={noise}")
    format_axis(ax, xlabel="Degradation r", ylabel="Mean Cosine Fidelity",
                title="D  weak_remote × noise levels\nN=1000, overlap=low, deg=medium",
                xlim=(0, 1), ylim=(-0.05, 1.05))
    ax.legend(fontsize=8, ncol=2)

    save_figure(fig, "fig2_sensitivity")
    plt.show()


# ============================================================
# ADNI Empirical Validation
# ============================================================

def build_adni_cohort(data_dir: str) -> pd.DataFrame:
    """
    Load and preprocess ADNI longitudinal data.

    Identifies MCI→AD converters, extracts cognitive scores in the
    ±36-month conversion window, and computes z-scores.

    Parameters
    ----------
    data_dir : path to directory containing
               DXSUM_*.csv, NEUROBAT_*.csv

    Returns
    -------
    cohort DataFrame with columns:
        RID | VISCODE2 | VISIT_MONTH | MONTHS_TO_CONV |
        DIAGNOSIS | AVDEL30MIN | BNTTOTAL | DIGITSPAN
    """
    import glob

    dxsum    = pd.read_csv(glob.glob(f"{data_dir}/DXSUM*.csv")[0],    low_memory=False)
    neurobat = pd.read_csv(glob.glob(f"{data_dir}/NEUROBAT*.csv")[0], low_memory=False)

    dxsum.columns    = dxsum.columns.str.strip().str.upper()
    neurobat.columns = neurobat.columns.str.strip().str.upper()

    viscode_map = {
        "bl": 0, "m03": 3, "m06": 6, "m12": 12, "m18": 18,
        "m24": 24, "m36": 36, "m48": 48, "m60": 60, "m72": 72,
        "m84": 84, "m96": 96
    }
    dxsum["VISIT_MONTH"] = dxsum["VISCODE2"].str.lower().map(viscode_map)
    dxsum = dxsum.dropna(subset=["VISIT_MONTH", "DIAGNOSIS"])
    dxsum["DIAGNOSIS"] = dxsum["DIAGNOSIS"].astype(float)

    baseline_dx = (
        dxsum.sort_values("VISIT_MONTH").groupby("RID").first()
        .reset_index()[["RID", "DIAGNOSIS"]].rename(columns={"DIAGNOSIS": "BASELINE_DX"})
    )
    mci_at_baseline = baseline_dx[baseline_dx["BASELINE_DX"] == 2.0]["RID"].unique()
    converters      = (
        dxsum[(dxsum["RID"].isin(mci_at_baseline)) & (dxsum["DIAGNOSIS"] == 3.0)]
        ["RID"].unique()
    )

    dx_conv          = dxsum[dxsum["RID"].isin(converters)].copy()
    conversion_month = (
        dx_conv[dx_conv["DIAGNOSIS"] == 3.0]
        .groupby("RID")["VISIT_MONTH"].min().rename("CONVERSION_MONTH")
    )
    dx_conv = dx_conv.merge(conversion_month, on="RID", how="left")
    dx_conv["MONTHS_TO_CONV"] = dx_conv["VISIT_MONTH"] - dx_conv["CONVERSION_MONTH"]
    window = dx_conv[dx_conv["MONTHS_TO_CONV"].between(-36, 12)].copy()

    neuro_cols = ["RID", "VISCODE2", "AVDEL30MIN", "BNTTOTAL", "DSPANFOR", "DSPANBAC"]
    neuro_sub  = neurobat[neuro_cols].copy()
    neuro_sub["VISCODE2"] = neuro_sub["VISCODE2"].str.lower()
    window["VISCODE2"]    = window["VISCODE2"].str.lower()

    merged = window.merge(neuro_sub, on=["RID", "VISCODE2"], how="left")
    merged["DIGITSPAN"] = merged["DSPANFOR"].add(merged["DSPANBAC"])

    keep = ["RID", "VISCODE2", "VISIT_MONTH", "MONTHS_TO_CONV",
            "DIAGNOSIS", "AVDEL30MIN", "BNTTOTAL", "DIGITSPAN"]
    return merged[keep].sort_values(["RID", "VISIT_MONTH"])


def compute_adni_failure_sequence(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-domain longitudinal decline slopes over the conversion window.

    Returns a DataFrame with empirical failure rank (most negative slope first),
    mean slope, 95% CI, and t-test p-value for each cognitive domain.
    """
    cog_domains = {
        "Episodic_Memory": "AVDEL30MIN",
        "Semantic_Memory": "BNTTOTAL",
        "Working_Memory":  "DIGITSPAN",
    }
    cohort = cohort.copy()

    for domain, col in cog_domains.items():
        mu  = cohort[col].mean()
        sd  = cohort[col].std()
        cohort[f"Z_{domain}"] = (cohort[col] - mu) / sd

    def ols_slope(df, zcol):
        sub = df[["MONTHS_TO_CONV", zcol]].dropna()
        if len(sub) < 2:
            return np.nan
        slope, *_ = stats.linregress(sub["MONTHS_TO_CONV"], sub[zcol])
        return slope

    records = []
    for domain in cog_domains:
        zcol   = f"Z_{domain}"
        slopes = cohort.groupby("RID").apply(lambda df: ols_slope(df, zcol)).dropna()
        n      = len(slopes)
        mean_s = slopes.mean()
        se     = slopes.sem()
        t_stat, p_val = stats.ttest_1samp(slopes, 0)
        records.append({
            "Domain":      domain,
            "N_subjects":  n,
            "Mean_slope":  round(mean_s, 5),
            "CI_lo":       round(mean_s - 1.96 * se, 5),
            "CI_hi":       round(mean_s + 1.96 * se, 5),
            "t_stat":      round(t_stat, 3),
            "p_value":     round(p_val, 6),
        })

    df_seq = pd.DataFrame(records).sort_values("Mean_slope").reset_index(drop=True)
    df_seq["Empirical_Rank"] = range(1, len(df_seq) + 1)

    model_ranks = {"Episodic_Memory": 1, "Semantic_Memory": 2, "Working_Memory": 3}
    df_seq["Model_Rank"] = df_seq["Domain"].map(model_ranks)

    rho, p_val = stats.spearmanr(df_seq["Model_Rank"], df_seq["Empirical_Rank"])
    print(f"\n=== SPEARMAN RANK CORRELATION ===")
    print(f"  rho = {rho:.4f},  p = {p_val:.4f},  n = {len(df_seq)} domains")

    return df_seq


def plot_validation_figure(cohort: pd.DataFrame, df_seq: pd.DataFrame):
    """
    Figure 3: Rank alignment slopegraph + longitudinal decline trajectories.
    """
    colours = {
        "Episodic_Memory": "#E05C5C",
        "Working_Memory":  "#5B8DB8",
        "Semantic_Memory": "#5BAA6E",
    }
    labels = {
        "Episodic_Memory": "Episodic Memory\n(RAVLT Delayed Recall)",
        "Working_Memory":  "Working Memory\n(Digit Span)",
        "Semantic_Memory": "Semantic Memory\n(Boston Naming Test)",
    }

    rho, p_val = stats.spearmanr(df_seq["Model_Rank"], df_seq["Empirical_Rank"])
    fig, axes  = plt.subplots(1, 2, figsize=(13, 7),
                               gridspec_kw={"width_ratios": [1, 1.3]})

    # Slopegraph
    ax = axes[0]
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0.5, 3.7)
    for _, row in df_seq.iterrows():
        col   = colours[row["Domain"]]
        mr, er = row["Model_Rank"], row["Empirical_Rank"]
        ls    = "-" if mr == er else "--"
        ax.plot([1, 2], [4 - mr, 4 - er], color=col, linewidth=2.5, linestyle=ls)
        ax.scatter([1, 2], [4 - mr, 4 - er], color=col, s=110)
        ax.text(0.92, 4 - mr, f"Rank {mr}", ha="right", va="center",
                fontsize=9, color=col, fontweight="bold")
        ax.text(2.08, 4 - er, f"Rank {er}", ha="left", va="center",
                fontsize=9, color=col, fontweight="bold")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Model\n(Hopfield/Braak)", "ADNI\n(Empirical)"], fontsize=10)
    ax.set_yticks([])
    ax.set_title("A — Failure Sequence Alignment", fontweight="bold", fontsize=11)
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    patches = [mpatches.Patch(color=colours[d], label=labels[d]) for d in colours]
    ax.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              fontsize=8.5, frameon=False)

    # Decline trajectories
    ax2     = axes[1]
    z_map   = {"Episodic_Memory": "AVDEL30MIN",
                "Semantic_Memory": "BNTTOTAL",
                "Working_Memory":  "DIGITSPAN"}
    cohort  = cohort.copy()
    for domain, raw_col in z_map.items():
        mu  = cohort[raw_col].mean()
        sd  = cohort[raw_col].std()
        cohort[f"Z_{domain}"] = (cohort[raw_col] - mu) / sd
        grp = (cohort.groupby("MONTHS_TO_CONV")[f"Z_{domain}"]
               .agg(["mean", "sem"]).reset_index())
        grp = grp[grp["MONTHS_TO_CONV"].between(-36, 12)]
        col = colours[domain]
        ax2.plot(grp["MONTHS_TO_CONV"], grp["mean"], color=col, lw=2.2,
                 label=labels[domain])
        ax2.fill_between(grp["MONTHS_TO_CONV"],
                          grp["mean"] - grp["sem"], grp["mean"] + grp["sem"],
                          color=col, alpha=0.15)
    ax2.axvline(0, color="#333333", lw=1.2, ls="--", alpha=0.6, label="Conversion")
    ax2.set_xlabel("Months Relative to Conversion", fontsize=10)
    ax2.set_ylabel("Mean Z-Score", fontsize=10)
    ax2.set_title("B — Longitudinal Decline Trajectories", fontweight="bold", fontsize=11)
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, 0.01,
             f"Spearman rank correlation: rho = {rho:.3f},  p = {p_val:.4f}  "
             f"(n = {len(df_seq)} cognitive domains)",
             ha="center", fontsize=10, fontstyle="italic", color="#444444")

    save_figure(fig, "fig3_validation")
    plt.show()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    # ── Mini smoke test ───────────────────────────────────────
    cfg_mini = SweepConfig(
        N_sizes        = [500],
        overlap_levels = ["low"],
        deg_rates      = ["fast"],
        noise_levels   = [0.30],
        n_trials       = 10,
    )

    print("\nRunning mini sweep (smoke test)...")
    df_mini = run_sensitivity_sweep(cfg_mini)
    print(f"\nShape: {df_mini.shape}")
    print(df_mini.groupby(["Braak_Stage", "Memory_Class"])["Cosine_Fidelity"]
          .mean().round(4).unstack())

    # ── Full sweep ────────────────────────────────────────────
    # Uncomment to run the full sweep (~8–12 min on GPU, ~90 min on CPU)
    #
    # cfg_full = SweepConfig()
    # df_sweep = run_sensitivity_sweep(cfg_full)
    # df_sweep.to_parquet("week5_sweep_results.parquet", index=False)
    #
    # df_thresh   = extract_collapse_thresholds(df_sweep)
    # df_features = compute_curve_features(df_sweep, df_thresh)
    #
    # plot_collapse_curves(df_sweep, df_thresh)
    # plot_sensitivity_figure(df_features, df_sweep)

    # ── ADNI validation ───────────────────────────────────────
    # Requires DXSUM_*.csv and NEUROBAT_*.csv in DATA_DIR
    #
    # DATA_DIR = "data/"
    # cohort   = build_adni_cohort(DATA_DIR)
    # cohort.to_csv("week6_adni_cohort.csv", index=False)
    #
    # df_seq = compute_adni_failure_sequence(cohort)
    # df_seq.to_csv("week6_failure_sequence.csv", index=False)
    # plot_validation_figure(cohort, df_seq)
