"""Mean and spread of results over seeds (runs made with --seed).

Reads experiments/results/baselines_s<seed>.csv and fl_<mode>_s<seed>.csv and writes

    experiments/results/seeds_summary.csv          headline rows (centralized, pooled local-only,
                                                   pooled FL): mean / std / min / max per metric
    experiments/results/seeds_per_client.csv       test PR-AUC per client: mean / std
    experiments/results/seeds_paired.csv           key comparisons, paired by seed: mean difference
                                                   in test PR-AUC with a 95% CI, how many seeds A
                                                   wins, and a paired t-test

Usage:
    python scripts/summarize_seeds.py
    python scripts/summarize_seeds.py --config configs/ablation_signal0.yaml   # that config's results_dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
METRICS = ["pr_auc", "roc_auc", "precision", "recall", "f1", "fpr"]
# (A, B): does A beat B? Setups are named fl_<mode> or <setup>_<model> (local_xgboost, ...).
PAIRS = [("fl_hybrid", "fl_zero_filled"), ("fl_hybrid", "fl_encoder_head"), ("fl_hybrid", "fl_naive"),
         ("fl_hybrid", "local_mlp"), ("fl_hybrid", "local_xgboost"), ("centralized_mlp", "fl_hybrid")]


def load_runs(results: Path = RESULTS) -> pd.DataFrame:
    files = sorted(results.glob("baselines_s*.csv")) + sorted(results.glob("fl_*_s*.csv"))
    files = [f for f in files if not f.stem.endswith("_by_pattern")]
    if not files:
        sys.exit(f"No per-seed results in {results}. Run run_baselines.py / run_fl.py with --seed.")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    headline = runs[runs["client"].isin(["all", "pooled"])]
    agg = headline.groupby(["setup", "client", "model"])[METRICS].agg(["mean", "std", "min", "max"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.insert(0, "seeds", headline.groupby(["setup", "client", "model"])["seed"].nunique())
    agg = agg.reset_index().sort_values("pr_auc_mean", ascending=False)

    per_client = runs[~runs["client"].isin(["all", "pooled"])]
    pc = per_client.groupby(["setup", "model", "client"])["pr_auc"].agg(["mean", "std"]).unstack("client")
    pc.columns = [f"{c}_{s}" for s, c in pc.columns]
    return agg, pc.reset_index()


def paired(runs: pd.DataFrame, pairs=PAIRS, metric: str = "pr_auc") -> pd.DataFrame:
    """A minus B on the headline (pooled / centralized) rows, seed by seed: both setups ran on the
    same data with the same seed, so the differences cancel what the seed does to both."""
    from scipy import stats

    head = runs[runs["client"].isin(["all", "pooled"])]
    names = [s if s.startswith("fl_") else f"{s}_{m}" for s, m in zip(head["setup"], head["model"])]
    wide = head.assign(name=names).pivot_table(index="seed", columns="name", values=metric)
    rows = []
    for a, b in pairs:
        if a not in wide or b not in wide:
            continue
        both = wide[[a, b]].dropna()
        d = both[a] - both[b]
        n = len(d)
        if n < 2:
            continue
        half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        rows.append({"a": a, "b": b, "metric": metric, "seeds": n, "a_mean": both[a].mean(), "b_mean": both[b].mean(),
                     "mean_diff": d.mean(), "ci95_low": d.mean() - half, "ci95_high": d.mean() + half,
                     "a_wins": int((d > 0).sum()), "p_value": float(stats.ttest_rel(both[a], both[b]).pvalue)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from aml_fl.utils import load_config, results_dir

    out = results_dir(load_config(ROOT / args.config), ROOT)
    runs = load_runs(out)
    agg, pc = summarize(runs)
    pairs = paired(runs)
    agg.to_csv(out / "seeds_summary.csv", index=False)
    pc.to_csv(out / "seeds_per_client.csv", index=False)
    pairs.to_csv(out / "seeds_paired.csv", index=False)
    print("| Setup | Model | Seeds | Test PR-AUC (mean ± std) | F1 | Recall | FPR |")
    print("|---|---|---|---|---|---|---|")
    for r in agg.itertuples():
        setup = r.setup if r.client == "all" else f"{r.setup} (pooled)"
        print(f"| {setup} | {r.model} | {r.seeds} | {r.pr_auc_mean:.3f} ± {r.pr_auc_std:.3f} | "
              f"{r.f1_mean:.3f} | {r.recall_mean:.3f} | {r.fpr_mean:.4f} |")
    if len(pairs):
        print("\n| A vs B (test PR-AUC) | Seeds | A − B, mean [95% CI] | A better on | p (paired t-test) |")
        print("|---|---|---|---|---|")
        for r in pairs.itertuples():
            print(f"| {r.a} vs {r.b} | {r.seeds} | {r.mean_diff:+.3f} [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}] | "
                  f"{r.a_wins}/{r.seeds} seeds | {r.p_value:.3f} |")
    print(f"\nSaved {out / 'seeds_summary.csv'}, {out / 'seeds_per_client.csv'} and {out / 'seeds_paired.csv'}")


if __name__ == "__main__":
    main()
