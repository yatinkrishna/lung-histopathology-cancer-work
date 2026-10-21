# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:33:00 2026

@author: Pooja
"""

# -*- coding: utf-8 -*-
"""
Statistical analysis + figures for the 5-fold U-Net segmentation results.

Reads what the training script already produced:
    cv_results_per_fold.csv   (per-fold final metrics)
    cv_summary.json           (mean/std/CI per metric, already computed)
    history_fold{0..4}.csv    (per-epoch training curves)

Produces, all saved under RESULTS_DIR/figures and RESULTS_DIR/stats:
    1. Extended descriptive statistics (mean, median, SD, IQR, 95% CI, 95%
       bootstrap CI) for every metric              -> stats/descriptive_stats.csv
    2. Shapiro-Wilk normality test per metric       -> stats/normality_tests.csv
    3. Pairwise metric comparison (paired Wilcoxon
       signed-rank + Cohen's d)                     -> stats/pairwise_comparisons.csv
    4. Pearson + Spearman correlation matrices
       between metrics across folds                 -> stats/correlation_pearson.csv,
                                                         stats/correlation_spearman.csv

    Fig 1: bar chart, mean +/- 95% CI, all metrics
    Fig 2: box + individual-fold strip plot, all metrics
    Fig 3: training curves (val Dice and train loss) per fold, epoch-wise
    Fig 4: correlation heatmap between metrics
    Fig 5: bootstrap sampling distribution for Dice and IoU, with CI shading
    Fig 6: Q-Q plot for Dice (+ Shapiro-Wilk result annotated)

NOTE on n=5 folds: with only 5 data points, normality tests, Wilcoxon tests
and bootstrap CIs all have limited power / are somewhat illustrative rather
than airtight. They are still what reviewers expect to see reported
alongside mean +/- std, so this script reports them with that caveat baked
into the printed output.
"""

# ============================================================
# CONFIG - edit these lines
# ============================================================
RESULTS_DIR = r"E:\ignite tool kit he segmentation\Data\unet_results"
METRICS     = ["dice", "iou", "prec", "rec", "acc"]   # must match columns in cv_results_per_fold.csv
N_FOLDS     = 5
N_BOOTSTRAP = 10000
SEED        = 42
DPI         = 300
# ============================================================

import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(SEED)

RESULTS_DIR = Path(RESULTS_DIR)
FIG_DIR     = RESULTS_DIR / "figures"
STATS_DIR   = RESULTS_DIR / "stats"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
PALETTE = sns.color_palette("Set2", n_colors=max(len(METRICS), N_FOLDS))


# ------------------------------------------------------------------
# 0. Load everything
# ------------------------------------------------------------------
def load_data():
    per_fold = pd.read_csv(RESULTS_DIR / "cv_results_per_fold.csv")

    with open(RESULTS_DIR / "cv_summary.json") as f:
        summary = json.load(f)

    histories = []
    for fold in range(N_FOLDS):
        hpath = RESULTS_DIR / f"history_fold{fold}.csv"
        if hpath.is_file():
            h = pd.read_csv(hpath)
            h["fold"] = fold
            histories.append(h)
    history_df = pd.concat(histories, ignore_index=True) if histories else None

    return per_fold, summary, history_df


# ------------------------------------------------------------------
# 1. Extended descriptive statistics
# ------------------------------------------------------------------
def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    boot_means = np.array([
        rng.choice(values, size=len(values), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return lo, hi, boot_means


def descriptive_stats(per_fold):
    rows = []
    boot_dists = {}
    for m in METRICS:
        vals = per_fold[m].values
        mean = vals.mean()
        median = np.median(vals)
        std = vals.std(ddof=1)
        q1, q3 = np.percentile(vals, [25, 75])
        t_ci = stats.t.interval(0.95, len(vals) - 1, loc=mean, scale=stats.sem(vals))
        b_lo, b_hi, boot_dist = bootstrap_ci(vals)
        boot_dists[m] = boot_dist

        rows.append({
            "metric": m, "mean": mean, "median": median, "std": std,
            "iqr_low": q1, "iqr_high": q3,
            "t_ci_low": t_ci[0], "t_ci_high": t_ci[1],
            "bootstrap_ci_low": b_lo, "bootstrap_ci_high": b_hi,
            "min": vals.min(), "max": vals.max(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "descriptive_stats.csv", index=False)
    print("\n=== Extended descriptive statistics ===")
    print(df.round(4).to_string(index=False))
    return df, boot_dists


# ------------------------------------------------------------------
# 2. Normality testing (Shapiro-Wilk)
# ------------------------------------------------------------------
def normality_tests(per_fold):
    rows = []
    for m in METRICS:
        vals = per_fold[m].values
        stat, p = stats.shapiro(vals)
        rows.append({"metric": m, "shapiro_stat": stat, "p_value": p,
                      "normal_at_0.05": p > 0.05})
    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "normality_tests.csv", index=False)
    print("\n=== Shapiro-Wilk normality tests (n=%d per metric, low power) ===" % len(per_fold))
    print(df.round(4).to_string(index=False))
    return df


# ------------------------------------------------------------------
# 3. Pairwise metric comparisons (paired Wilcoxon + Cohen's d)
# ------------------------------------------------------------------
def cohens_d_paired(a, b):
    diff = np.asarray(a) - np.asarray(b)
    return diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan


def pairwise_comparisons(per_fold):
    rows = []
    for m1, m2 in combinations(METRICS, 2):
        a, b = per_fold[m1].values, per_fold[m2].values
        try:
            stat, p = stats.wilcoxon(a, b)
        except ValueError:
            stat, p = np.nan, np.nan
        d = cohens_d_paired(a, b)
        rows.append({
            "metric_a": m1, "metric_b": m2,
            "wilcoxon_stat": stat, "p_value": p,
            "cohens_d": d,
        })
    df = pd.DataFrame(rows)
    df.to_csv(STATS_DIR / "pairwise_comparisons.csv", index=False)
    print("\n=== Pairwise metric comparisons (paired Wilcoxon, n=%d folds) ===" % len(per_fold))
    print(df.round(4).to_string(index=False))
    return df


# ------------------------------------------------------------------
# 4. Correlation matrices
# ------------------------------------------------------------------
def correlation_matrices(per_fold):
    sub = per_fold[METRICS]
    pearson = sub.corr(method="pearson")
    spearman = sub.corr(method="spearman")
    pearson.to_csv(STATS_DIR / "correlation_pearson.csv")
    spearman.to_csv(STATS_DIR / "correlation_spearman.csv")
    return pearson, spearman


# ------------------------------------------------------------------
# Figures
# ------------------------------------------------------------------
def fig1_bar_ci(desc_df):
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(desc_df))
    means = desc_df["mean"].values
    err_low = means - desc_df["t_ci_low"].values
    err_high = desc_df["t_ci_high"].values - means
    ax.bar(x, means, yerr=[err_low, err_high], capsize=6,
           color=PALETTE[:len(desc_df)], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in desc_df["metric"]])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean ± 95% CI across 5 folds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_bar_mean_ci.png", dpi=DPI)
    plt.close(fig)


def fig2_box_strip(per_fold):
    long_df = per_fold.melt(id_vars=["fold"], value_vars=METRICS,
                             var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=long_df, x="metric", y="score", ax=ax,
                palette=PALETTE, showfliers=False)
    sns.stripplot(data=long_df, x="metric", y="score", ax=ax,
                  color="black", size=7, jitter=True, alpha=0.7)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_title("Per-fold distribution by metric (5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_box_strip.png", dpi=DPI)
    plt.close(fig)


def fig3_training_curves(history_df):
    if history_df is None:
        print("No history_fold*.csv files found - skipping Fig 3.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for fold in sorted(history_df["fold"].unique()):
        sub = history_df[history_df["fold"] == fold]
        axes[0].plot(sub["epoch"], sub["dice"], color=PALETTE[fold],
                     alpha=0.5, label=f"Fold {fold}")
        axes[1].plot(sub["epoch"], sub["train_loss"], color=PALETTE[fold],
                     alpha=0.5, label=f"Fold {fold}")

    mean_dice = history_df.groupby("epoch")["dice"].mean()
    mean_loss = history_df.groupby("epoch")["train_loss"].mean()
    axes[0].plot(mean_dice.index, mean_dice.values, color="black",
                 linewidth=3, label="Mean")
    axes[1].plot(mean_loss.index, mean_loss.values, color="black",
                 linewidth=3, label="Mean")

    axes[0].set_title("Validation Dice per epoch")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Dice")
    axes[1].set_title("Training loss per epoch")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[0].legend(fontsize=9); axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_training_curves.png", dpi=DPI)
    plt.close(fig)


def fig4_correlation_heatmap(pearson):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(pearson, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                square=True, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation between metrics across folds")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_correlation_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig5_bootstrap_distributions(boot_dists):
    metrics_to_show = [m for m in ["dice", "iou"] if m in boot_dists]
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(7 * len(metrics_to_show), 5))
    if len(metrics_to_show) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics_to_show):
        dist = boot_dists[m]
        lo, hi = np.percentile(dist, [2.5, 97.5])
        sns.histplot(dist, kde=True, ax=ax, color=PALETTE[METRICS.index(m)])
        ax.axvline(lo, color="red", linestyle="--", label="95% CI")
        ax.axvline(hi, color="red", linestyle="--")
        ax.axvline(dist.mean(), color="black", label="Bootstrap mean")
        ax.set_title(f"Bootstrap distribution of mean {m.upper()}\n({N_BOOTSTRAP} resamples)")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_bootstrap_distributions.png", dpi=DPI)
    plt.close(fig)


def fig6_qq_plot(per_fold, normality_df):
    fig, ax = plt.subplots(figsize=(6, 6))
    vals = per_fold["dice"].values
    stats.probplot(vals, dist="norm", plot=ax)
    row = normality_df[normality_df["metric"] == "dice"].iloc[0]
    ax.set_title(f"Q-Q plot: Dice\nShapiro-Wilk p = {row['p_value']:.3f}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_qq_dice.png", dpi=DPI)
    plt.close(fig)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    per_fold, summary, history_df = load_data()

    desc_df, boot_dists = descriptive_stats(per_fold)
    normality_df = normality_tests(per_fold)
    pairwise_df = pairwise_comparisons(per_fold)
    pearson, spearman = correlation_matrices(per_fold)

    fig1_bar_ci(desc_df)
    fig2_box_strip(per_fold)
    fig3_training_curves(history_df)
    fig4_correlation_heatmap(pearson)
    fig5_bootstrap_distributions(boot_dists)
    fig6_qq_plot(per_fold, normality_df)

    print(f"\nAll stats CSVs saved to  -> {STATS_DIR.resolve()}")
    print(f"All figures (PNG, {DPI} dpi) saved to -> {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()