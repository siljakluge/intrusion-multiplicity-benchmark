"""Validation-selected predictive multiplicity and security failure audit."""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .datasets import load_prepared


def _short_tol(value: float) -> str:
    return f"eps_{value:.5f}".replace(".", "p")


def _pack_disagreement(left: np.ndarray, right: np.ndarray) -> int:
    packed_a = np.packbits(left.astype(bool))
    packed_b = np.packbits(right.astype(bool))
    return int(np.unpackbits(np.bitwise_xor(packed_a, packed_b)).sum())


def compute_disagreement(run_dir: str | Path, tolerance: float = 0.001, metric: str = "val_accuracy") -> Path:
    run_dir = Path(run_dir)
    if tolerance < 0:
        raise ValueError("Rashomon tolerance must not be negative")
    metrics = pd.read_csv(run_dir/"metrics.csv")
    if metric not in {"val_accuracy", "val_macro_f1"}:
        raise ValueError("Selection metric must be validation accuracy or validation macro-F1")
    maximum = metrics[metric].max()
    selected = metrics.loc[metrics[metric] >= maximum-tolerance].sort_values("model_id")
    model_ids = selected.model_id.tolist()
    if not model_ids:
        raise ValueError("No selected models")
    data = load_prepared(run_dir)
    predictions = []
    for model_id in model_ids:
        with np.load(run_dir/"predictions"/f"{model_id}.npz") as z:
            p = z["y_pred"].copy()
        if len(p) != len(data.y_test):
            raise ValueError("Prediction length is not equal to the test sample count")
        predictions.append(p)
    matrix = np.stack(predictions).astype(np.uint8)
    n_models, n_samples = matrix.shape
    p_attack = matrix.mean(axis=0)
    conflict = np.minimum(p_attack, 1-p_attack)
    unanimous_normal = p_attack == 0
    unanimous_attack = p_attack == 1
    contested = (p_attack > 0)&(p_attack < 1)
    entropy = -p_attack*np.log2(np.clip(p_attack,1e-12,1)) - (1-p_attack)*np.log2(np.clip(1-p_attack,1e-12,1))
    majority = (p_attack > .5).astype(np.int8)
    # Exact ties receive no forced majority-attack preference.
    ties = p_attack == .5
    majority[ties] = -1
    out = run_dir/"analysis"/_short_tol(tolerance)
    out.mkdir(parents=True,exist_ok=True)
    selected.assign(rashomon_metric=metric,rashomon_tolerance=tolerance,
                    rashomon_best_score=maximum).to_csv(out/"rashomon_models.csv",index=False)
    table = data.metadata_test.copy()
    table.insert(0,"sample_id",np.arange(n_samples))
    table["y_true"] = data.y_test
    table["predicted_attack_fraction"] = p_attack
    table["conflict_ratio"] = conflict
    table["is_conflict"] = contested
    table["vote_entropy"] = entropy
    table["majority_pred"] = majority
    table["consensus_state"] = np.select([unanimous_normal,unanimous_attack],
                                         ["unanimous_normal", "unanimous_attack"],default="contested")
    table["unanimous_false_negative"] = unanimous_normal & (data.y_test == 1)
    table["unanimous_false_positive"] = unanimous_attack & (data.y_test == 0)
    # Unseen attacks can only be identified if the type taxonomy is shared.
    seen = set(data.split_description.get("training_attack_types", []))
    table["unseen_attack_type"] = ((table.y_true==1)&~table.attack_type.astype(str).isin(seen))
    table.to_csv(out/"sample_disagreement.csv",index=False)

    rows = []
    for i,j in combinations(range(n_models),2):
        ndiff = _pack_disagreement(matrix[i],matrix[j])
        rows.append({"model_a":model_ids[i],"model_b":model_ids[j],
                     "disagreement_rate":ndiff/n_samples,"n_differing_samples":ndiff,
                     "comparison_block_a":selected.iloc[i].comparison_block,
                     "comparison_block_b":selected.iloc[j].comparison_block})
    pairs = pd.DataFrame(rows,columns=["model_a","model_b","disagreement_rate","n_differing_samples",
                                     "comparison_block_a","comparison_block_b"])
    pairs.to_csv(out/"disagreement_summary.csv",index=False)
    attack = table.loc[table.y_true==1]
    families = []
    for typ,group in table.groupby("attack_type", dropna=False):
        n_attack = int((group.y_true==1).sum())
        families.append({"attack_type":typ,"n_samples":len(group),"n_attacks":n_attack,
                         "n_unanimous_misses":int(group.unanimous_false_negative.sum()),
                         "fraction_unanimous_misses":float(group.unanimous_false_negative.mean()),
                         "fraction_contested":float(group.is_conflict.mean()),
                         "n_unseen_attack_type":int(group.unseen_attack_type.sum())})
    pd.DataFrame(families).to_csv(out/"security_by_attack_type.csv",index=False)
    summary = {"n_models_total":len(metrics), "n_models_selected":n_models,"n_test":n_samples,
               "validation_metric":metric,"rashomon_tolerance":tolerance,"best_validation_score":float(maximum),
               "ambiguity":float(contested.mean()),"mean_conflict_ratio":float(conflict.mean()),
               "mean_pairwise_disagreement":float(pairs.disagreement_rate.mean()) if len(pairs) else 0.,
               "max_pairwise_disagreement":float(pairs.disagreement_rate.max()) if len(pairs) else 0.,
               "unanimous_false_negatives":int(table.unanimous_false_negative.sum()),
               "unanimous_false_positives":int(table.unanimous_false_positive.sum()),
               "unseen_attack_instances":int(table.unseen_attack_type.sum()),
               "false_negative_rate_unanimous":float(attack.unanimous_false_negative.mean()) if len(attack) else None,
               "n_unanimous_normal":int(unanimous_normal.sum()), "n_unanimous_attack":int(unanimous_attack.sum()),
               "n_contested":int(contested.sum())}
    (out/"multiplicity_summary.json").write_text(json.dumps(summary,indent=2))
    pd.DataFrame([summary]).to_csv(out/"multiplicity_summary.csv",index=False)
    return out


def diagnose_shift(run_dir: str | Path, max_features: int = 200) -> Path:
    """Descriptive diagnostics; not a causal attribution of performance gap."""
    run_dir=Path(run_dir)
    data=load_prepared(run_dir)
    out=run_dir/"diagnostics"
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,x,y in [("train",data.x_train,data.y_train),
                      ("validation",data.x_val,data.y_val),("test",data.x_test,data.y_test)]:
        rows.append({"split":label,"n":len(y),"n_benign":int((y==0).sum()),
                     "n_attack":int((y==1).sum()),"attack_fraction":float((y==1).mean())})
    pd.DataFrame(rows).to_csv(out/"class_distribution.csv",index=False)
    known=set(data.split_description.get("training_attack_types",[]))
    attack_type=data.metadata_test.attack_type.astype(str)
    unseen=(data.y_test==1)&(~attack_type.isin(known).to_numpy())
    pd.DataFrame([{"n_attacks":int((data.y_test==1).sum()),"n_unseen_attacks":int(unseen.sum()),
                  "unseen_attack_fraction":float(unseen.sum()/max((data.y_test==1).sum(),1))}]).to_csv(out/"unseen_types.csv",index=False)
    shifts=[]
    rng=np.random.default_rng(42)
    for idx in range(min(len(data.feature_names),max_features)):
        a=data.x_train[:,idx];b=data.x_test[:,idx]
        aa=rng.choice(a,size=min(len(a),5000),replace=False)
        bb=rng.choice(b,size=min(len(b),5000),replace=False)
        ks=ks_2samp(aa,bb).statistic
        shifts.append({"feature":data.feature_names[idx],"ks_statistic":float(ks),
                       "mean_train":float(aa.mean()),"mean_test":float(bb.mean())})
    pd.DataFrame(shifts).sort_values("ks_statistic",ascending=False).to_csv(out/"feature_shift.csv",index=False)
    return out


def aggregate_runs(run_dirs: list[str | Path], tolerance: float = .001, out_dir: str | Path = "reports") -> Path:
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for path in map(Path,run_dirs):
        report=path/"analysis"/_short_tol(tolerance)/"multiplicity_summary.json"
        if not report.exists():
            raise FileNotFoundError(f"Analyze {path} at tolerance {tolerance} first")
        record=json.loads(report.read_text());record["run_dir"]=str(path)
        rows.append(record)
    df=pd.DataFrame(rows)
    df.to_csv(out/"origin_results.csv",index=False)
    summary={}
    for col in df.select_dtypes(include="number").columns:
        summary[col]={"mean":float(df[col].mean()), "sample_sd":float(df[col].std(ddof=1)) if len(df)>1 else None}
    (out/"aggregate.json").write_text(json.dumps({"n_origins":len(df),"statistics":summary,
                        "note":"Origins are replication units; model pairs share models and test cases."},indent=2))
    # Distinguish persistence from independence; only calculate where sample identities match.
    samples=[]
    for path in map(Path,run_dirs):
        f=path/"analysis"/_short_tol(tolerance)/"sample_disagreement.csv"
        samples.append(pd.read_csv(f, usecols=["source_file","source_row","y_true","is_conflict","unanimous_false_negative"]))
    fingerprints=[pd.util.hash_pandas_object(s[["source_file","source_row","y_true"]],index=False).sum() for s in samples]
    if len(set(fingerprints))==1 and all(samples[0][["source_file","source_row","y_true"]].equals(s[["source_file","source_row","y_true"]]) for s in samples[1:]):
        persistence=samples[0][["source_file","source_row","y_true"]].copy()
        persistence["n_origins_conflict"]=sum(s.is_conflict.astype(int) for s in samples)
        persistence["persistent_unanimous_miss"]=np.logical_and.reduce([s.unanimous_false_negative.to_numpy(dtype=bool) for s in samples])
        persistence.to_csv(out/"cross_origin_persistence.csv",index=False)
    else:
        (out/"cross_origin_persistence_NOT_COMPUTED.txt").write_text(
            "Test samples differ across origin runs. Persistence is not meaningful across different test sets.\n")
    return out
