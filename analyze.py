#!/usr/bin/env python3
"""
Multi-seed analysis of ECG self-supervised models (PTB-XL super-classes, linear probing).

Reads the saved test-set predictions in  results/<model>/seed<N>/<fraction>/<name>.pkl  and produces:
  analysis/seed_values.csv      one row per model / label fraction / seed / metric
  analysis/summary.csv|.md      mean, sd over seeds, median, min-max, test-set sd
  analysis/ranking.csv          how often each model ordering appears across seeds
  analysis/tests.csv            paired hierarchical-bootstrap tests between models
  analysis/fig1_*.png|pdf       metric vs. label fraction, every seed shown
  analysis/fig2_*.png|pdf       model differences with 95% confidence intervals

Usage:  python analyze.py [--results results] [--out analysis] [--n-boot 2000]

Method (short)
  * Metrics: macro AUROC and macro AUPRC over the 5 PTB-XL super-classes, recomputed from the predictions.
  * Seed variability: mean / sd over seeds (ddof=1). Seeds change the label subsample and the classifier
    (and, for Random, the encoder initialisation).
  * Test-set variability: patients of the test set are resampled (bootstrap).
  * Model comparison: all models share the same test patients and, for a given seed, the same training
    subset, so differences are computed per seed and per bootstrap replicate (paired).
    Hierarchical bootstrap: each replicate resamples test patients AND seeds, so the interval reflects both
    sources of variability. Two-sided p-value from the bootstrap distribution; Holm correction across the
    9 comparisons of each metric.
"""
import argparse
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# folder name -> (display name, file stem inside the folder)
MODELS = {
    "merl": ("MERL", "MERL"),
    "clocs": ("CLOCS", "CLOCS"),
    "random_init": ("Random init", "Random_500Hz_Z_score_none_No_Bandpass_Resnet18"),
}
FRACTIONS = {"0.01": "1%", "0.1": "10%", "1.0": "100%"}
METRICS = {"auroc": "macro AUROC", "auprc": "macro AUPRC"}
PAIRS = [("merl", "random_init"), ("clocs", "random_init"), ("merl", "clocs")]


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def load_predictions(results_dir):
    """Return y_true (n, 5) and preds[model][fraction][seed] = y_pred (n, 5)."""
    preds, y_ref = {}, None
    for m, (_, stem) in MODELS.items():
        preds[m] = {}
        for frac in FRACTIONS:
            preds[m][frac] = {}
            for d in sorted(glob.glob(f"{results_dir}/{m}/seed*/{frac}")):
                seed = int(re.search(r"seed(\d+)", d).group(1))
                with open(f"{d}/{stem}.pkl", "rb") as f:
                    obj = pickle.load(f)
                if y_ref is None:
                    y_ref = obj["y_true"]
                if not np.array_equal(obj["y_true"], y_ref):
                    sys.exit(f"Test labels differ for {m} seed {seed} fraction {frac}: pairing is not valid.")
                preds[m][frac][seed] = obj["y_pred"]
    seeds = {tuple(sorted(preds[m][f])) for m in preds for f in FRACTIONS}
    if len(seeds) != 1:
        sys.exit(f"Models do not all have the same seeds: {seeds}")
    return y_ref, preds, list(seeds.pop())


# ----------------------------------------------------------------------------------------------
# Metrics (point estimate and fast bootstrap)
# ----------------------------------------------------------------------------------------------
def macro_boot(y, P, W):
    """Macro AUROC and AUPRC for B bootstrap weightings.

    y: (n, C) binary labels, P: (n, C) scores, W: (B, n) integer resampling counts
    (number of times each patient is drawn). Equivalent to resampling patients with replacement and
    calling sklearn, but vectorised. A patient drawn k times is a group of k tied copies with the same
    label, which is handled exactly (AUROC: no positive-negative ties; AUPRC: one threshold per group).
    """
    B, n = W.shape
    auc = np.zeros(B)
    ap = np.zeros(B)
    for c in range(y.shape[1]):
        order = np.argsort(P[:, c], kind="stable")          # ascending score
        pos = y[order, c]
        neg = 1.0 - pos
        w = W[:, order]
        wp, wn = w * pos, w * neg
        cum_neg_below = np.cumsum(wn, axis=1) - wn
        auc += (wp * cum_neg_below).sum(1) / (wp.sum(1) * wn.sum(1))
        # AUPRC: walk from the highest score down
        wr, posr = w[:, ::-1], pos[::-1]
        wpr = wr * posr
        # max(.,1): where nobody has been drawn yet, wpr is 0 too, so the term is 0 (avoids 0/0)
        ap += (wpr * np.cumsum(wpr, axis=1) / np.maximum(np.cumsum(wr, axis=1), 1.0)).sum(1) / wpr.sum(1)
    return auc / y.shape[1], ap / y.shape[1]


def macro_point(y, P):
    auc = np.mean([roc_auc_score(y[:, c], P[:, c]) for c in range(y.shape[1])])
    ap = np.mean([average_precision_score(y[:, c], P[:, c]) for c in range(y.shape[1])])
    return auc, ap


def self_check(y, P, W, rng):
    """The vectorised bootstrap must agree with sklearn on explicit resamples."""
    auc_b, ap_b = macro_boot(y, P, W)
    for b in rng.choice(W.shape[0], 3, replace=False):
        idx = np.repeat(np.arange(W.shape[1]), W[b].astype(int))
        a, p = macro_point(y[idx], P[idx])
        assert abs(a - auc_b[b]) < 1e-9 and abs(p - ap_b[b]) < 1e-9, "bootstrap self-check failed"
    a, p = macro_point(y, P)
    a1, p1 = macro_boot(y, P, np.ones((1, len(y))))
    assert abs(a - a1[0]) < 1e-9 and abs(p - p1[0]) < 1e-9, "point-estimate self-check failed"


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def hierarchical_diff(Ma, Mb, rng):
    """Ma, Mb: (S, B) bootstrap metric per seed. Returns B values of the seed-averaged difference,
    resampling seeds with replacement inside every bootstrap replicate."""
    D = Ma - Mb
    S, B = D.shape
    idx = rng.integers(0, S, size=(B, S))
    return D[idx, np.arange(B)[:, None]].mean(1)


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"merl": "#2a78d6", "clocs": "#eb6834", "random_init": "#1baf7a"}   # validated categorical slots 1-3
MARKERS = {"merl": "o", "clocs": "s", "random_init": "^"}                       # shape = second encoding


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_metric_vs_fraction(seed_df, out, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), facecolor=SURFACE)
    xs = np.arange(len(FRACTIONS))
    offs = {"merl": -0.14, "clocs": 0.0, "random_init": 0.14}
    for ax, (mk, mlabel) in zip(axes, METRICS.items()):
        style_axes(ax)
        for m, (name, _) in MODELS.items():
            means = []
            for i, frac in enumerate(FRACTIONS):
                v = seed_df[(seed_df.model == m) & (seed_df.fraction == frac) & (seed_df.metric == mk)].sort_values("seed")["value"].values
                jitter = np.linspace(-0.045, 0.045, len(v))
                ax.scatter(xs[i] + offs[m] + jitter, v, s=14, color=COLORS[m], alpha=0.45, marker=MARKERS[m],
                           linewidths=0, zorder=2)
                means.append(v.mean())
            ax.plot(xs + offs[m], means, color=COLORS[m], marker=MARKERS[m], markersize=7, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v} labels" for v in FRACTIONS.values()])
        ax.set_xlim(-0.5, len(xs) - 0.5)
        ax.set_ylabel(mlabel, color=INK2, fontsize=9)
        ax.set_title(mlabel, loc="left", color=INK, fontsize=11, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK, loc="lower right")
    fig.text(0.01, 0.01, f"Lines: mean over {len(seeds)} seeds ({seeds[0]}-{seeds[-1]}). Faint markers: individual seeds. PTB-XL super-classes, linear probing.",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig1_metric_vs_label_fraction.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def fig_differences(tests, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2), facecolor=SURFACE, sharey=True)
    for ax, (mk, mlabel) in zip(axes, METRICS.items()):
        style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        y, yticks, ylabels = 0, [], []
        for (a, b) in PAIRS:
            for frac, flabel in FRACTIONS.items():
                r = tests[(tests.metric == mk) & (tests.A == a) & (tests.B == b) & (tests.fraction == frac)].iloc[0]
                sig = r.p_holm < 0.05
                ax.errorbar(r.mean_diff, y, xerr=[[r.mean_diff - r.ci_low], [r.ci_high - r.mean_diff]], fmt="o", color=INK,
                            ecolor=INK2, elinewidth=1.6, capsize=3, markersize=7,
                            markerfacecolor=INK if sig else SURFACE, markeredgewidth=1.6, zorder=3)
                yticks.append(y)
                ylabels.append(f"{MODELS[a][0]} - {MODELS[b][0]}  ({flabel} labels)")
                y += 1
            y += 0.7
        ax.axvline(0, color=INK2, linewidth=1, zorder=1)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=9, color=INK)
        ax.set_ylim(y - 0.7, -1)      # first comparison at the top (axis shared by both panels)
        ax.set_xlabel(f"Difference in {mlabel} (95% CI)", color=INK2, fontsize=9)
        ax.set_title(mlabel, loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Filled = significant after Holm correction (alpha = 0.05); hollow = not significant. "
             "Interval: hierarchical bootstrap over test patients and seeds.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}/fig2_model_differences.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--rng", type=int, default=0, help="seed of the bootstrap itself")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.rng)

    y, preds, seeds = load_predictions(args.results)
    n, S, B = len(y), len(seeds), args.n_boot
    print(f"{n} test patients, seeds {seeds}, {B} bootstrap replicates")

    # one set of patient resamples shared by every model, fraction and seed (=> paired comparisons)
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=B).astype(np.float64)
    self_check(y, preds["merl"]["1.0"][seeds[0]], W, rng)
    print("self-check against scikit-learn: OK")

    # ---- point estimates and bootstrap distributions
    rows, boot = [], {}          # boot[(model, frac, metric)] -> (S, B)
    for m in MODELS:
        for frac in FRACTIONS:
            A, P = [], []
            for s in seeds:
                a, p = macro_point(y, preds[m][frac][s])
                rows += [dict(model=m, fraction=frac, seed=s, metric="auroc", value=a),
                         dict(model=m, fraction=frac, seed=s, metric="auprc", value=p)]
                ab, pb = macro_boot(y, preds[m][frac][s], W)
                A.append(ab)
                P.append(pb)
            boot[(m, frac, "auroc")] = np.array(A)
            boot[(m, frac, "auprc")] = np.array(P)
    seed_df = pd.DataFrame(rows)
    seed_df.to_csv(f"{args.out}/seed_values.csv", index=False)

    # ---- summary table
    summ = []
    for (m, frac, mk), g in seed_df.groupby(["model", "fraction", "metric"]):
        v = g.sort_values("seed")["value"].values
        test_sd = boot[(m, frac, mk)].std(axis=1, ddof=1).mean()
        summ.append(dict(model=MODELS[m][0], fraction=FRACTIONS[frac], metric=mk, n_seeds=len(v), mean=v.mean(),
                         sd_seeds=v.std(ddof=1), median=np.median(v), min=v.min(), max=v.max(), sd_test_set=test_sd,
                         per_seed=" ".join(f"{x:.4f}" for x in v)))
    summ = pd.DataFrame(summ)
    order_m = [MODELS[m][0] for m in MODELS]
    summ["_m"] = summ.model.map({k: i for i, k in enumerate(order_m)})
    summ["_f"] = summ.fraction.map({v: i for i, v in enumerate(FRACTIONS.values())})
    summ = summ.sort_values(["metric", "_f", "_m"]).drop(columns=["_m", "_f"])
    summ.to_csv(f"{args.out}/summary.csv", index=False, float_format="%.5f")
    with open(f"{args.out}/summary.md", "w") as f:
        for mk, mlabel in METRICS.items():
            f.write(f"### {mlabel} (mean +/- sd over {S} seeds [min-max]; sd of test-set bootstrap in brackets)\n\n")
            f.write("| Labels | " + " | ".join(order_m) + " |\n|---|" + "---|" * len(order_m) + "\n")
            for flabel in FRACTIONS.values():
                cells = []
                for name in order_m:
                    r = summ[(summ.metric == mk) & (summ.fraction == flabel) & (summ.model == name)].iloc[0]
                    cells.append(f"{r['mean']:.3f} +/- {r.sd_seeds:.3f} [{r['min']:.3f}-{r['max']:.3f}] ({r.sd_test_set:.3f})")
                f.write(f"| {flabel} | " + " | ".join(cells) + " |\n")
            f.write("\n")

    # ---- ranking consistency
    rk = []
    for mk in METRICS:
        for frac, flabel in FRACTIONS.items():
            counts = {}
            for s in seeds:
                vals = {m: seed_df[(seed_df.model == m) & (seed_df.fraction == frac) & (seed_df.seed == s) &
                                   (seed_df.metric == mk)]["value"].iloc[0] for m in MODELS}
                ordering = " > ".join(MODELS[m][0] for m in sorted(vals, key=vals.get, reverse=True))
                counts[ordering] = counts.get(ordering, 0) + 1
            for o, c in sorted(counts.items(), key=lambda t: -t[1]):
                rk.append(dict(metric=mk, fraction=flabel, ordering=o, n_seeds=c, of=S))
    pd.DataFrame(rk).to_csv(f"{args.out}/ranking.csv", index=False)

    # ---- paired hierarchical bootstrap tests
    trows = []
    for mk in METRICS:
        block = []
        for (a, b) in PAIRS:
            for frac in FRACTIONS:
                Ma, Mb = boot[(a, frac, mk)], boot[(b, frac, mk)]
                d = hierarchical_diff(Ma, Mb, rng)
                obs = (seed_df[(seed_df.model == a) & (seed_df.fraction == frac) & (seed_df.metric == mk)].sort_values("seed")["value"].values
                       - seed_df[(seed_df.model == b) & (seed_df.fraction == frac) & (seed_df.metric == mk)].sort_values("seed")["value"].values)
                p = min(1.0, 2 * (min((d <= 0).sum(), (d >= 0).sum()) + 1) / (B + 1))
                block.append(dict(metric=mk, fraction=frac, A=a, B=b, mean_diff=obs.mean(),
                                  ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5), p_raw=p,
                                  seeds_A_better=f"{int((obs > 0).sum())}/{S}"))
        p_adj = holm([r["p_raw"] for r in block])
        for r, pa in zip(block, p_adj):
            r["p_holm"] = pa
        trows += block
    tests = pd.DataFrame(trows)
    out_t = tests.copy()
    out_t["A"] = out_t.A.map(lambda k: MODELS[k][0])
    out_t["B"] = out_t.B.map(lambda k: MODELS[k][0])
    out_t["fraction"] = out_t.fraction.map(FRACTIONS)
    out_t.to_csv(f"{args.out}/tests.csv", index=False, float_format="%.5f")

    if not args.no_figures:
        fig_metric_vs_fraction(seed_df, args.out, seeds)
        fig_differences(tests, args.out)

    # ---- console report
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n" + open(f"{args.out}/summary.md").read())
    print(out_t[out_t.metric == "auroc"].drop(columns="metric").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nFiles written to {args.out}/")


if __name__ == "__main__":
    main()
