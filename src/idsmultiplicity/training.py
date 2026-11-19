"""Factorized 63-model design derived from the original mmcyber experimental plan.

Per origin: baseline(1) + initialization(20) + bootstrap(20) + architecture/
activation(20) + training subset(2), with validation-only model selection.
"""
from __future__ import annotations

import json
import random
import platform
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .datasets import _csv_paths, load_prepared, prepare_dataset, save_prepared


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str, dropout: float):
        super().__init__()
        choices = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}
        if activation not in choices:
            raise ValueError(f"Unknown activation: {activation}")
        layers = []
        previous = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(previous, width), choices[activation](), nn.BatchNorm1d(width), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


def device_from_name(name: str) -> torch.device:
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else ("cpu" if name == "auto" else name))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _stream_seed(origin: int, stream: int, local: int = 0) -> int:
    if not 0 <= local < 10_000:
        raise ValueError("Comparison seeds must be in [0, 10,000)")
    return origin * 100_000 + stream * 10_000 + local


def plan_models(training: dict, origin: int) -> list[dict]:
    base = list(training.get("hidden_dims", [128, 64]))
    base_activation = training.get("activation", "relu")
    seeds = [int(s) for s in training.get("comparison_seeds", list(range(2, 22)))]
    variants = training.get("architecture_variants", [[32,16],[64,32],[96,48],[128,64],[160,80],[192,96],[256,128]])
    activations = training.get("activation_variants", ["relu", "tanh", "leaky_relu"])
    block = []
    def add(group, train_seed, data_seed, dims=base, act=base_activation, fraction=1., strategy="stratified"):
        identifier = f"origin{origin}_{group}_ts{train_seed}_ds{data_seed}_arch{'x'.join(map(str,dims))}_{act}_frac{fraction:g}"
        block.append({"model_id": identifier, "origin_seed": origin, "comparison_block": group,
                      "train_seed": train_seed, "data_seed": data_seed, "hidden_dims": list(dims),
                      "activation": act, "subset_fraction": float(fraction), "subset_strategy": strategy})
    bseed, bdata = _stream_seed(origin, 0), _stream_seed(origin, 1)
    add("baseline", bseed, bdata)
    for s in seeds:
        add("seed", _stream_seed(origin, 2, s), bdata)
    for s in seeds:
        add("bootstrap", _stream_seed(origin, 3, s), _stream_seed(origin, 4, s), strategy="bootstrap")
    for dims in variants:
        for act in activations:
            if list(dims) == base and act == base_activation:
                continue
            add("architecture_activation", _stream_seed(origin, 5), bdata, dims=list(dims), act=act)
    for fraction in training.get("subset_fractions", [0.75, 0.5]):
        if float(fraction) != 1.:
            add("subset", bseed, bdata, fraction=float(fraction))
    if len({x["model_id"] for x in block}) != len(block):
        raise ValueError("Duplicate model IDs in experiment plan")
    return block


def _subset(y: np.ndarray, spec: dict) -> np.ndarray:
    rng = np.random.default_rng(spec["data_seed"])
    if spec["subset_strategy"] == "bootstrap":
        return rng.choice(len(y), len(y), replace=True)
    frac = spec["subset_fraction"]
    if frac >= 1:
        return np.arange(len(y))
    result = []
    for cls in np.unique(y):
        indices = np.flatnonzero(y == cls)
        result.append(rng.choice(indices, max(1, round(len(indices) * frac)), replace=False))
    return rng.permutation(np.concatenate(result))


def _batches(x, y=None, size=256, shuffle=False, seed=0):
    arrays = [torch.from_numpy(x)] + ([] if y is None else [torch.from_numpy(y)])
    return DataLoader(TensorDataset(*arrays), batch_size=size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed))


def predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 2048):
    model.eval()
    pieces = []
    with torch.no_grad():
        for (batch,) in _batches(x, size=batch_size):
            pieces.append(torch.softmax(model(batch.to(device)), dim=1).cpu().numpy())
    probs = np.concatenate(pieces)
    return probs.argmax(axis=1), probs


def _metrics(y, pred, probs, prefix=""):
    return {prefix+"accuracy": float(accuracy_score(y, pred)),
            prefix+"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            prefix+"log_loss": float(log_loss(y, probs, labels=[0,1]))}


def train_model(data, spec: dict, training: dict, run_dir: Path):
    set_seed(spec["train_seed"])
    idx = _subset(data.y_train, spec)
    x, y = data.x_train[idx], data.y_train[idx]
    device = device_from_name(training.get("device", "auto"))
    model = MLPClassifier(data.x_train.shape[1], 2, spec["hidden_dims"], spec["activation"], float(training.get("dropout", .2))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training.get("learning_rate", .001)),
                                  weight_decay=float(training.get("weight_decay", .0001)))
    criterion = nn.CrossEntropyLoss()
    best, state, stale = float("inf"), None, 0
    size = int(training.get("batch_size", 256))
    for epoch in range(int(training.get("epochs", 20))):
        model.train()
        for xb,yb in _batches(x,y,size,True, spec["train_seed"] + epoch):
            xb,yb = xb.to(device),yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb),yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss_sum = 0.
        with torch.no_grad():
            for xb,yb in _batches(data.x_val,data.y_val,size):
                xb,yb = xb.to(device),yb.to(device)
                val_loss_sum += criterion(model(xb),yb).item() * len(yb)
        val_loss = val_loss_sum/len(data.y_val)
        if val_loss < best - 1e-9:
            best, state, stale = val_loss, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= int(training.get("patience",5)):
                break
    if state is None:
        raise RuntimeError("No checkpoint found")
    model.load_state_dict(state)
    val_pred, val_prob = predict(model, data.x_val, device, size)
    test_pred, test_prob = predict(model, data.x_test, device, size)
    checkpoint_path = run_dir / "models" / f"{spec['model_id']}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict":state, "input_dim":data.x_train.shape[1],"output_dim":2,
                "hidden_dims":spec["hidden_dims"], "activation":spec["activation"],
                "dropout":float(training.get("dropout", .2))}, checkpoint_path)
    row = {**spec, "subset_size":len(idx), "best_val_loss":best, "epochs_ran":epoch+1,
           "model_path":str(checkpoint_path), **_metrics(data.y_val,val_pred,val_prob,"val_"),
           **_metrics(data.y_test,test_pred,test_prob)}
    pred_path = run_dir / "predictions" / f"{spec['model_id']}.npz"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(pred_path, y_pred=test_pred.astype(np.int8), probs=test_prob.astype(np.float32))
    return row


def load_model(path: Path, device: torch.device | None = None):
    device = device or torch.device("cpu")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = MLPClassifier(checkpoint["input_dim"], checkpoint["output_dim"],
                          checkpoint["hidden_dims"], checkpoint["activation"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def run_training(cfg: dict, origin: int, *, max_models: int | None = None) -> Path:
    run_dir = Path(cfg.get("run_dir", "runs/{dataset_name}/origin_seed{origin_seed}").format(
        dataset_name=cfg["dataset"]["dataset"], origin_seed=origin))
    run_dir.mkdir(parents=True, exist_ok=True)
    split_seed = _stream_seed(origin, 8, int(cfg["dataset"].get("split_seed", 42)))
    resolved = {**cfg, "origin_seed":origin, "split_seed":split_seed}
    config_path = run_dir / "config.resolved.json"
    if config_path.exists() and json.loads(config_path.read_text()) != resolved:
        raise ValueError(f"Config mismatch in existing {run_dir}; use a new run directory")
    config_path.write_text(json.dumps(resolved, indent=2))
    if (run_dir / "prepared_data.npz").exists():
        old_manifest = json.loads((run_dir / "split_manifest.json").read_text())
        dcfg = cfg["dataset"]
        patterns = dcfg.get("train_files", []) + dcfg.get("test_files", []) if dcfg.get("train_files") else dcfg.get("files", ["**/*.csv"])
        discovered = {str(p) for p in _csv_paths(Path(dcfg["root"]).expanduser(), patterns)}
        recorded = {item["path"] for item in old_manifest.get("raw_files", [])}
        if discovered != recorded:
            raise ValueError("Dataset source file set changed since the run cache was created")
        for item in old_manifest.get("raw_files", []):
            raw_path = Path(item["path"])
            if (not raw_path.exists() or raw_path.stat().st_size != item["size"] or
                    raw_path.stat().st_mtime_ns != item["mtime_ns"]):
                raise ValueError(f"Raw input changed since cached run: {raw_path}. Use a new run directory.")
        data = load_prepared(run_dir)
    else:
        data = prepare_dataset(cfg["dataset"], run_dir, split_seed)
        save_prepared(data, run_dir)
    env = {"python": sys.version, "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__}
    try:
        import shap, sklearn
        env.update({"shap": shap.__version__, "sklearn": sklearn.__version__})
    except ImportError:
        pass
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parent, check=False)
    env["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    (run_dir / "environment.json").write_text(json.dumps(env, indent=2))
    plan = plan_models(cfg["training"], origin)
    pd.DataFrame([{**row, "hidden_dims": str(row["hidden_dims"])} for row in plan]).to_csv(run_dir/"training_plan.csv",index=False)
    to_run = plan[:max_models] if max_models else plan
    results = []
    for spec in tqdm(to_run, desc=f"origin-{origin}"):
        result_path = run_dir/"model_results"/f"{spec['model_id']}.json"
        prediction_file = run_dir/"predictions"/f"{spec['model_id']}.npz"
        checkpoint_file = run_dir/"models"/f"{spec['model_id']}.pt"
        if all(p.exists() for p in [result_path,prediction_file,checkpoint_file]):
            result = json.loads(result_path.read_text())
        else:
            result = train_model(data,spec,cfg["training"],run_dir)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(result, indent=2))
        results.append(result)
        pd.DataFrame([{**r, "hidden_dims":str(r["hidden_dims"])} for r in results]).to_csv(run_dir/"metrics.csv",index=False)
    return run_dir
