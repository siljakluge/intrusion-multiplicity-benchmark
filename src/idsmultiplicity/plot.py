"""Publication-ready vector plots from computed CSV/JSON outputs (no synthetic data)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import _short_tol


def _finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_run(run_dir: str | Path, tolerance: float = .001) -> Path:
    run_dir=Path(run_dir)
    folder=run_dir/"analysis"/_short_tol(tolerance)
    target=folder/"figures";target.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(run_dir/"metrics.csv")
    fig,ax=plt.subplots(figsize=(5.2,3.2))
    xx=np.arange(2)
    vals=[metrics.val_accuracy.mean()*100,metrics.accuracy.mean()*100]
    ax.bar(xx,vals,width=.55)
    ax.set_xticks(xx,["Validation","Test"])
    ax.set_ylim(0,100);ax.set_ylabel("Mean accuracy (%)")
    _finish(fig,target/"validation_test_gap.pdf")

    sample=pd.read_csv(folder/"sample_disagreement.csv")
    groups=sample.groupby("consensus_state",sort=False).size()
    fig,ax=plt.subplots(figsize=(5.2,3.2))
    ax.barh(groups.index,groups.values)
    ax.set_xlabel("Test instances (n)")
    _finish(fig,target/"consensus_states.pdf")

    attacks=pd.read_csv(folder/"security_by_attack_type.csv")
    attacks=attacks.loc[attacks.n_attacks>=10].sort_values("fraction_unanimous_misses",ascending=False).head(12)
    fig,ax=plt.subplots(figsize=(6.2,4.4))
    ax.barh(attacks.attack_type.astype(str).iloc[::-1],attacks.fraction_unanimous_misses.iloc[::-1]*100)
    ax.set_xlabel("Unanimous misses (% of attack-type test samples)")
    _finish(fig,target/"unanimous_misses_by_attack_type.pdf")
    explanation_file=folder/"shap"/"local_explanation_samples.csv"
    if explanation_file.exists():
        local=pd.read_csv(explanation_file)
        names=["Unanimous","Contested"]
        groups=[local.loc[~local.is_conflict.astype(bool),"mean_local_shap_variance"].dropna(),
                local.loc[local.is_conflict.astype(bool),"mean_local_shap_variance"].dropna()]
        if all(len(v)>0 for v in groups):
            fig,ax=plt.subplots(figsize=(5.2,3.3))
            ax.boxplot(groups,showfliers=False)
            ax.set_xticks([1, 2], names)
            ax.set_ylabel("Mean variance of SHAP attack-logit attributions")
            _finish(fig,target/"local_shap_agreement.pdf")
    return target
