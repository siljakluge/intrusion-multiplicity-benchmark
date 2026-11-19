"""SHAP agreement and local variability on fixed test instances.

All models are explained with the same background and same class-1 logit.
Never compare explanations of different target classes as if they were the same.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analysis import _short_tol
from .datasets import load_prepared
from .training import load_model, device_from_name


def _choose_samples(table: pd.DataFrame, max_explain: int, seed: int):
    rng=np.random.default_rng(seed)
    conflict=table.loc[table.is_conflict.astype(bool),"sample_id"].to_numpy(dtype=int)
    stable=table.loc[~table.is_conflict.astype(bool),"sample_id"].to_numpy(dtype=int)
    # Prefer balanced samples, but use all classes when one group is small.
    n_conf=min(len(conflict),max_explain//2)
    n_stable=min(len(stable),max_explain-n_conf)
    extra=max_explain-(n_conf+n_stable)
    if extra:
        n_conf+=min(extra,len(conflict)-n_conf)
        extra=max_explain-(n_conf+n_stable)
        n_stable+=min(extra,len(stable)-n_stable)
    ids=np.concatenate([rng.choice(conflict,n_conf,replace=False),rng.choice(stable,n_stable,replace=False)])
    return np.sort(ids)


def _attack_shap(values, n_samples, n_features):
    values=np.asarray(values)
    if values.shape==(n_samples,n_features,2):
        return values[:,:,1]
    if values.shape==(2,n_samples,n_features):
        return values[1]
    if values.shape==(n_samples,2,n_features):
        return values[:,1,:]
    # Binary single-output wrapper, when provided explicitly in the future.
    if values.shape==(n_samples,n_features):
        return values
    raise ValueError(f"Unrecognized SHAP shape {values.shape} for {n_samples}x{n_features}x2")


def _cosine_distance(a,b):
    denom=np.linalg.norm(a)*np.linalg.norm(b)
    return float(1-(a@b/denom)) if denom else 0.


def compute_shap(run_dir: str | Path, tolerance: float=.001, *, max_background: int=64,
                 max_explain: int=128, seed: int=42, device: str="cpu", top_k: int=20) -> Path:
    import shap
    import torch

    run_dir=Path(run_dir)
    analysis=run_dir/"analysis"/_short_tol(tolerance)
    selected=pd.read_csv(analysis/"rashomon_models.csv")
    samples=pd.read_csv(analysis/"sample_disagreement.csv")
    data=load_prepared(run_dir)
    if max_background<=0 or max_explain<=0:
        raise ValueError("SHAP sample sizes must be positive")
    rng=np.random.default_rng(seed)
    bg_ids=np.sort(rng.choice(len(data.y_train),min(max_background,len(data.y_train)),replace=False))
    sample_ids=_choose_samples(samples,max_explain,seed)
    if not len(sample_ids):
        raise ValueError("No test instances selected for SHAP")
    out=analysis/"shap"
    out.mkdir(parents=True,exist_ok=True)
    manifest={"selected_model_ids":sorted(selected.model_id.tolist()),"background_indices":bg_ids.tolist(),
              "explain_indices":sample_ids.tolist(),"shap_output":"attack class logit, class index 1",
              "n_features":len(data.feature_names),"feature_names":data.feature_names}
    manifest_path=out/"shap_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("SHAP selection changed. Use a fresh output directory or remove the previous SHAP outputs")
    manifest_path.write_text(json.dumps(manifest,indent=2))
    x_bg=torch.from_numpy(data.x_train[bg_ids])
    x_test=torch.from_numpy(data.x_test[sample_ids])
    selected_ids=sorted(selected.model_id.tolist())
    attribution=[]
    importance=[]
    for model_id in selected_ids:
        path=out/f"{model_id}.npz"
        if path.exists():
            with np.load(path) as z:
                values=z["attribution"].copy()
        else:
            torch_device=device_from_name(device)
            model=load_model(run_dir/"models"/f"{model_id}.pt",torch_device)
            explainer=shap.DeepExplainer(model,x_bg.to(torch_device))
            values=_attack_shap(explainer.shap_values(x_test.to(torch_device)),len(sample_ids),len(data.feature_names))
            values=np.asarray(values,dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite SHAP values for {model_id}")
            np.savez_compressed(path,attribution=values)
            del explainer,model
        if values.shape!=(len(sample_ids),len(data.feature_names)):
            raise ValueError(f"Invalid cached SHAP attribution shape for {model_id}")
        attribution.append(values)
        importance.append(np.abs(values).mean(axis=0))
    attribution=np.stack(attribution)
    importance=np.stack(importance)
    pd.DataFrame(importance.T,columns=selected_ids,index=data.feature_names).rename_axis("feature").to_csv(out/"global_importance.csv")
    rank=np.argsort(-importance,axis=1)
    rows=[]
    pairs=pd.read_csv(analysis/"disagreement_summary.csv")
    disagreement={(r.model_a,r.model_b):r.disagreement_rate for r in pairs.itertuples()}
    for i,j in combinations(range(len(selected_ids)),2):
        features_i=set(rank[i,:min(top_k,len(data.feature_names))])
        features_j=set(rank[j,:min(top_k,len(data.feature_names))])
        rows.append({"model_a":selected_ids[i],"model_b":selected_ids[j],
                     "cosine_distance_full_features":_cosine_distance(importance[i],importance[j]),
                     "top_k_jaccard":len(features_i&features_j)/max(1,len(features_i|features_j)),
                     "prediction_disagreement":disagreement.get((selected_ids[i],selected_ids[j]),np.nan)})
    pd.DataFrame(rows).to_csv(out/"global_explanation_pairs.csv",index=False)
    # Variance and sign instability across models, on the same samples/features.
    variance=np.var(attribution,axis=0,ddof=0)
    sign_fraction=(attribution>0).mean(axis=0)
    instability=np.minimum(sign_fraction,1-sign_fraction)
    table=samples.set_index("sample_id").loc[sample_ids].reset_index()
    table["mean_local_shap_variance"]=variance.mean(axis=1)
    table["mean_local_sign_instability"]=instability.mean(axis=1)
    table["mean_local_shap_range"]=(attribution.max(axis=0)-attribution.min(axis=0)).mean(axis=1)
    table.to_csv(out/"local_explanation_samples.csv",index=False)
    if table.conflict_ratio.nunique()>1 and table.mean_local_shap_variance.nunique()>1:
        correlation=float(spearmanr(table.conflict_ratio,table.mean_local_shap_variance).statistic)
    else:
        correlation=None
    stable=table.loc[~table.is_conflict.astype(bool)]
    contested=table.loc[table.is_conflict.astype(bool)]
    info={"n_models":len(selected_ids),"n_explained_samples":len(sample_ids),
          "n_conflict":len(contested),"n_unanimous":len(stable),
          "mean_local_variance_conflict":float(contested.mean_local_shap_variance.mean()) if len(contested) else None,
          "mean_local_variance_unanimous":float(stable.mean_local_shap_variance.mean()) if len(stable) else None,
          "mean_local_sign_instability":float(table.mean_local_sign_instability.mean()),
          "spearman_conflict_vs_local_variance":correlation,
          "mean_global_cosine_distance":float(np.mean([r["cosine_distance_full_features"] for r in rows])) if rows else None,
          "mean_global_topk_jaccard":float(np.mean([r["top_k_jaccard"] for r in rows])) if rows else None,
          "note":"Descriptive statistics; not an independent sample of model pairs or random test examples. SHAP output is class-1 logit."}
    (out/"explanation_summary.json").write_text(json.dumps(info,indent=2))
    return out
