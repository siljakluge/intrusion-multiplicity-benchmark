"""Command-line interface for end-to-end paper reproduction."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .analysis import aggregate_runs, compute_disagreement, diagnose_shift
from .explain import compute_shap
from .training import plan_models, run_training
from .plot import plot_run


def read_config(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg,dict) or "dataset" not in cfg or "training" not in cfg:
        raise ValueError("Config needs 'dataset' and 'training' mappings")
    return cfg


def main():
    p=argparse.ArgumentParser(prog="idsmultiplicity")
    sub=p.add_subparsers(dest="action",required=True)
    q=sub.add_parser("plan",help="Show and optionally save model design without training")
    q.add_argument("--config",required=True)
    q.add_argument("--origin",type=int,default=1)
    q=sub.add_parser("train",help="Train or resume one origin")
    q.add_argument("--config",required=True)
    q.add_argument("--origin",type=int,default=1)
    q.add_argument("--max-models",type=int)
    q=sub.add_parser("run-all",help="Train and audit all configured origins")
    q.add_argument("--config",required=True)
    q.add_argument("--tolerance",type=float,default=.001)
    q.add_argument("--with-shap",action="store_true")
    q.add_argument("--max-models",type=int)
    q=sub.add_parser("disagree",help="Validation-select Rashomon models and compute test conflict")
    q.add_argument("--run-dir",required=True)
    q.add_argument("--tolerance",type=float,default=.001)
    q.add_argument("--metric",choices=["val_accuracy","val_macro_f1"],default="val_accuracy")
    q=sub.add_parser("diagnose",help="Diagnose feature-shift and unseen test attack types")
    q.add_argument("--run-dir",required=True)
    q=sub.add_parser("explain",help="SHAP audit for validation-selected models")
    q.add_argument("--run-dir",required=True)
    q.add_argument("--tolerance",type=float,default=.001)
    q.add_argument("--background",type=int,default=64)
    q.add_argument("--samples",type=int,default=128)
    q.add_argument("--device",default="cpu")
    q=sub.add_parser("plot",help="Export vector PDF figures from computed results")
    q.add_argument("--run-dir",required=True)
    q.add_argument("--tolerance",type=float,default=.001)
    q=sub.add_parser("aggregate",help="Aggregate across origins without treating model pairs as independent")
    q.add_argument("--run-dirs",nargs="+",required=True)
    q.add_argument("--tolerance",type=float,default=.001)
    q.add_argument("--out-dir",default="reports")
    a=p.parse_args()
    if a.action=="plan":
        cfg=read_config(a.config)
        from collections import Counter
        plan=plan_models(cfg["training"],a.origin)
        print(f"Models per origin: {len(plan)}; blocks: {dict(Counter(s['comparison_block'] for s in plan))}")
    elif a.action=="train":
        print(run_training(read_config(a.config),a.origin,max_models=a.max_models))
    elif a.action=="run-all":
        cfg=read_config(a.config)
        run_dirs=[]
        for origin in cfg["training"].get("origin_seeds",list(range(1,11))):
            out=run_training(cfg,int(origin),max_models=a.max_models)
            compute_disagreement(out,a.tolerance)
            diagnose_shift(out)
            if a.with_shap:
                compute_shap(out,a.tolerance)
            run_dirs.append(out)
        print(aggregate_runs(run_dirs,a.tolerance,Path(cfg.get("reports_dir","reports"))/cfg["dataset"]["dataset"]))
    elif a.action=="disagree":
        print(compute_disagreement(a.run_dir,a.tolerance,a.metric))
    elif a.action=="diagnose":
        print(diagnose_shift(a.run_dir))
    elif a.action=="explain":
        print(compute_shap(a.run_dir,a.tolerance,max_background=a.background,
                           max_explain=a.samples,device=a.device))
    elif a.action=="plot":
        print(plot_run(a.run_dir,a.tolerance))
    elif a.action=="aggregate":
        print(aggregate_runs(a.run_dirs,a.tolerance,a.out_dir))


if __name__=="__main__":
    main()
