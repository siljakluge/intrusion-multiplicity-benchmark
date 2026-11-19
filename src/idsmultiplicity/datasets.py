"""Local-only IDS CSV loaders, leakage guards, and reproducible split/preprocessing.

The data files are not distributed with this repository. Provider download URLs
are documented in docs/DATASETS.md, not fetched implicitly.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NSL_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

# These are never valid predictor features in our paper's binary task.
LEAKY_NAMES = {
    "label", "class", "target", "attack", "attack_cat", "attack_category",
    "attack_type", "attack_class", "label_orig", "is_attack", "difficulty",
    "id", "unnamed: 0", "flow_id", "flow id", "timestamp", "time_stamp",
    "src_ip", "dst_ip", "srcip", "dstip", "src ip", "dst ip", "source ip",
    "destination ip", "source_ip", "destination_ip", "src_mac", "dst_mac",
}


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    metadata_train: pd.DataFrame
    metadata_val: pd.DataFrame
    metadata_test: pd.DataFrame
    preprocessor: Any
    split_description: dict


def _find_column(columns, candidates: list[str], *, required: bool = True) -> str | None:
    available = list(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    norm = {str(name).strip().casefold(): name for name in available}
    for candidate in candidates:
        if candidate.strip().casefold() in norm:
            return norm[candidate.strip().casefold()]
    if required:
        raise ValueError(f"None of columns {candidates!r} found. Available: {available[:30]!r}")
    return None


def _csv_paths(root: Path, patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(paths)


def _csv_frame(path: Path, cfg: dict, seed: int) -> pd.DataFrame:
    read_kwargs = dict(low_memory=False)
    if cfg.get("dataset") == "nsl_kdd":
        read_kwargs["names"] = NSL_COLUMNS
        read_kwargs["header"] = None
    cap = cfg.get("max_rows_per_file")
    if cap is None:
        frame = pd.read_csv(path, **read_kwargs)
        frame["__source_row"] = np.arange(len(frame), dtype=np.int64)
    else:
        cap = int(cap)
        if cap < 2:
            raise ValueError("max_rows_per_file must be >= 2")
        # Uniform reservoir by independent random priorities; bounded memory.
        rng = np.random.default_rng(seed)
        chosen = []
        scores = np.empty(0, dtype=np.float64)
        offset = 0
        for chunk in pd.read_csv(path, chunksize=int(cfg.get("chunk_rows", 30_000)), **read_kwargs):
            chunk["__source_row"] = np.arange(offset, offset + len(chunk), dtype=np.int64)
            offset += len(chunk)
            priorities = rng.random(len(chunk))
            if len(chunk) > cap:
                keep = np.argpartition(priorities, cap - 1)[:cap]
                chunk = chunk.iloc[keep].copy()
                priorities = priorities[keep]
            chosen.append(chunk)
            scores = np.concatenate([scores, priorities])
            if len(scores) > cap:
                combined = pd.concat(chosen, ignore_index=True)
                keep = np.argpartition(scores, cap - 1)[:cap]
                chosen = [combined.iloc[keep].copy()]
                scores = scores[keep]
        if not chosen:
            raise ValueError(f"No rows in {path}")
        frame = pd.concat(chosen, ignore_index=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError(f"Duplicate column names in {path}")
    frame["__source_file"] = path.name
    return frame


def _read_files(paths: list[Path], cfg: dict, seed: int) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError("No CSV files matched. See docs/DATASETS.md for folder layout.")
    tables = []
    for i, path in enumerate(paths):
        frame = _csv_frame(path, cfg, seed + 97 * i)
        frame["__source_file"] = path.relative_to(Path(cfg["root"]).expanduser()).as_posix()
        tables.append(frame)
    columns = [set(table.columns) - {"__source_file"} for table in tables]
    if any(s != columns[0] for s in columns[1:]):
        raise ValueError("CSV files have different schemas. Choose one protocol/subset or harmonize CSV columns first.")
    return pd.concat(tables, ignore_index=True)


def _to_binary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frame = df.copy()
    candidates = [cfg["label_column"]] if cfg.get("label_column") else cfg.get("label_candidates", ["Label", "label", "class"])
    label_col = _find_column(frame.columns, candidates)
    benign = {str(v).strip().casefold() for v in cfg.get("benign_labels", ["normal", "benign", "benigntraffic", "0"])}
    series = frame[label_col]
    norm = series.astype(str).str.strip().str.casefold()
    if norm.isna().any() or series.isna().any():
        raise ValueError("Missing labels cannot be interpreted as attacks")
    # When a canonical binary label exists, disallow unexpected values.
    if cfg.get("binary_label", False):
        if not set(norm.unique()).issubset({"0", "1"}):
            raise ValueError(f"Binary label {label_col!r} contains unexpected values: {norm.unique()[:10]}")
        frame["__target"] = norm.astype(int).astype(np.int64)
    else:
        frame["__target"] = (~norm.isin(benign)).astype(np.int64)
    type_col = _find_column(frame.columns, cfg.get("attack_type_candidates", ["attack_cat", "Attack", "Label", "label"]), required=False)
    if type_col is not None and type_col != label_col:
        frame["__attack_type"] = frame[type_col].astype(str).str.strip()
    elif not cfg.get("binary_label", False):
        frame["__attack_type"] = series.astype(str).str.strip()
    else:
        frame["__attack_type"] = np.where(frame["__target"].to_numpy() == 0, "Benign", "Attack")
    return frame


def _load_raw(cfg: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    root = Path(cfg["root"]).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}. Download it from official source (docs/DATASETS.md).")
    train_patterns = cfg.get("train_files")
    test_patterns = cfg.get("test_files")
    if bool(train_patterns) != bool(test_patterns):
        raise ValueError("train_files and test_files must both be provided for an official split")
    if train_patterns:
        train_paths, test_paths = _csv_paths(root, train_patterns), _csv_paths(root, test_patterns)
        if set(train_paths) & set(test_paths):
            raise ValueError("Train/test file patterns overlap")
        train_raw = _to_binary(_read_files(train_paths, cfg, seed), cfg)
        test_raw = _to_binary(_read_files(test_paths, cfg, seed + 100), cfg)
        return train_raw, test_raw, {"method": "official", "train_files": [str(p) for p in train_paths], "test_files": [str(p) for p in test_paths]}
    paths = _csv_paths(root, cfg.get("files", ["**/*.csv"]))
    raw = _to_binary(_read_files(paths, cfg, seed), cfg)
    return raw, None, {"method": cfg.get("split", "random"), "files": [str(p) for p in paths]}


def _split_data(full: pd.DataFrame, test: pd.DataFrame | None, cfg: dict, seed: int):
    test_fraction = float(cfg.get("test_fraction", 0.2))
    val_fraction = float(cfg.get("validation_fraction", 0.2))
    if not (0 < test_fraction < 0.5 and 0 < val_fraction < 0.5):
        raise ValueError("test_fraction and validation_fraction must be between 0 and 0.5")
    if test is not None:
        method = "official"
        trainval = full
    else:
        method = cfg.get("split", "random")
        if method == "random":
            trainval, test = train_test_split(full, test_size=test_fraction, random_state=int(cfg.get("test_split_seed", 42)),
                                              stratify=full["__target"])
        elif method == "temporal":
            col = _find_column(full.columns, [cfg.get("time_column", "Timestamp")])
            dates = pd.to_datetime(full[col], errors="coerce", format="mixed", utc=True)
            if dates.isna().any():
                raise ValueError("Temporal split has missing/unparseable timestamps: clean data or use explicit split")
            sorted_full = full.assign(__sort_time=dates).sort_values("__sort_time", kind="stable")
            cut = int(round(len(sorted_full) * (1 - test_fraction)))
            trainval, test = sorted_full.iloc[:cut], sorted_full.iloc[cut:]
        elif method == "group":
            group_col = _find_column(full.columns, [cfg["group_column"]])
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=int(cfg.get("test_split_seed", 42)))
            a, b = next(splitter.split(full, full["__target"], full[group_col]))
            trainval, test = full.iloc[a], full.iloc[b]
        else:
            raise ValueError(f"Unknown split method {method!r}")
    if method == "temporal":
        trainval = trainval.sort_values("__sort_time", kind="stable")
        cut = int(round(len(trainval) * (1 - val_fraction)))
        train, val = trainval.iloc[:cut], trainval.iloc[cut:]
    elif method == "group":
        group_col = _find_column(trainval.columns, [cfg["group_column"]])
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed + 17)
        a, b = next(splitter.split(trainval, trainval["__target"], trainval[group_col]))
        train, val = trainval.iloc[a], trainval.iloc[b]
    else:
        train, val = train_test_split(trainval, test_size=val_fraction, random_state=seed,
                                      stratify=trainval["__target"])
    for name, frame in [("train", train), ("validation", val), ("test", test)]:
        if frame["__target"].nunique() != 2:
            raise ValueError(f"{name} has only one class. Adjust split protocol / dataset sample.")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _feature_columns(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: dict):
    excluded = set(cfg.get("drop_columns", [])) | set(cfg.get("exclude_columns", []))
    excluded |= {k for k in train.columns if k.startswith("__")}
    excluded |= {k for k in train.columns if k.strip().casefold() in LEAKY_NAMES}
    excluded |= {cfg.get("label_column", "")}
    if cfg.get("time_column"):
        excluded.add(cfg["time_column"])
    if cfg.get("group_column"):
        excluded.add(cfg["group_column"])
    columns = [name for name in train.columns if name not in excluded]
    for c in columns:
        if c not in val or c not in test:
            raise ValueError(f"Feature {c!r} missing from val/test")
    if not columns:
        raise ValueError("No predictor features remain")
    return columns


def prepare_dataset(cfg: dict, run_dir: Path, split_seed: int) -> PreparedData:
    full, official_test, protocol = _load_raw(cfg, int(cfg.get("sampling_seed", 42)))
    train, val, test = _split_data(full, official_test, cfg, split_seed)
    columns = _feature_columns(train, val, test, cfg)
    configured = cfg.get("categorical_columns", [])
    categorical = [c for c in columns if c in configured or str(train[c].dtype) in {"object", "string", "category"}]
    numeric = [c for c in columns if c not in categorical]
    for frame in (train, val, test):
        for col in numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        for col in categorical:
            frame[col] = frame[col].astype("string").fillna("unknown").astype(str)
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    x_train = preprocessor.fit_transform(train[columns]).astype(np.float32)
    x_val = preprocessor.transform(val[columns]).astype(np.float32)
    x_test = preprocessor.transform(test[columns]).astype(np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    def meta(frame: pd.DataFrame):
        keep = [c for c in ["__attack_type", "__source_file", "__source_row", cfg.get("time_column"), cfg.get("group_column")] if c and c in frame]
        return frame[keep].rename(columns={"__attack_type":"attack_type","__source_file":"source_file", "__source_row":"source_row"}).copy()
    paths = protocol.get("files", []) + protocol.get("train_files", []) + protocol.get("test_files", [])
    protocol["raw_files"] = [{"path": str(path), "size": Path(path).stat().st_size,
                              "mtime_ns": Path(path).stat().st_mtime_ns} for path in paths]
    protocol.update({"seed": int(split_seed), "n_train": len(train), "n_val": len(val),
                     "n_test": len(test), "feature_columns": columns,
                     "categorical_columns": categorical, "numeric_columns": numeric,
                     "test_class_counts": pd.Series(test["__target"]).value_counts().to_dict(),
                     "training_attack_types": sorted(train.loc[train["__target"] == 1, "__attack_type"].dropna().astype(str).unique())})
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, run_dir / "preprocessor.joblib")
    data = PreparedData(x_train, x_val, x_test,
                        train["__target"].to_numpy(dtype=np.int64),
                        val["__target"].to_numpy(dtype=np.int64),
                        test["__target"].to_numpy(dtype=np.int64),
                        feature_names, ["normal", "attack"], meta(train), meta(val), meta(test),
                        preprocessor, protocol)
    return data


def save_prepared(data: PreparedData, run_dir: Path) -> None:
    path = run_dir / "prepared_data.npz"
    arrays = {k: getattr(data, k) for k in ("x_train", "x_val", "x_test", "y_train", "y_val", "y_test")}
    np.savez(path, **arrays)
    pd.Series(data.feature_names, name="feature").to_csv(run_dir / "feature_names.csv", index=False)
    data.metadata_train.to_csv(run_dir / "metadata_train.csv", index=False)
    data.metadata_val.to_csv(run_dir / "metadata_val.csv", index=False)
    data.metadata_test.to_csv(run_dir / "metadata_test.csv", index=False)
    (run_dir / "split_manifest.json").write_text(json.dumps(data.split_description, indent=2, default=str))
    digest = hashlib.sha256()
    for value in arrays.values():
        digest.update(np.ascontiguousarray(value).view(np.uint8))
    (run_dir / "dataset_fingerprint.json").write_text(json.dumps({"sha256":digest.hexdigest()}, indent=2))


def load_prepared(run_dir: Path) -> PreparedData:
    with np.load(run_dir / "prepared_data.npz") as source:
        arrays = {k: source[k].copy() for k in ("x_train", "x_val", "x_test", "y_train", "y_val", "y_test")}
    return PreparedData(**arrays,
                        feature_names=pd.read_csv(run_dir / "feature_names.csv")["feature"].tolist(),
                        class_names=["normal", "attack"],
                        metadata_train=pd.read_csv(run_dir / "metadata_train.csv"),
                        metadata_val=pd.read_csv(run_dir / "metadata_val.csv"),
                        metadata_test=pd.read_csv(run_dir / "metadata_test.csv"),
                        preprocessor=joblib.load(run_dir / "preprocessor.joblib"),
                        split_description=json.loads((run_dir / "split_manifest.json").read_text()))
