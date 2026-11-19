# Intrusion Detection Model Multiplicity Audit

**Reproducible Python research code for predictive, explanatory, and security-failure analyses of neural intrusion detectors.**

This project is a standalone successor to [`siljakluge/model-multiplicity-cyberdef`](https://github.com/siljakluge/model-multiplicity-cyberdef), focused exclusively on network intrusion detection. It reproduces the **factorized 63-MLP candidate design per origin** of the previous work, and adds dataset-specific loaders, model agreement/failure analysis, split auditing, class-1 SHAP comparisons, and cross-origin reporting.

> **Status: research-code candidate, not a validated submission artifact.** The unit tests and end-to-end synthetic-data smoke test pass. Real-data loaders are schema-aware but **have not been validated against every full third-party raw archive** in this deliverable. Reproduce at least one complete run and inspect the selected raw dataset's column names, labels, duplicate flows, and grouping before citing paper results.

## Install

Python 3.11+; for GPU installation follow the [official PyTorch selector](https://pytorch.org/get-started/locally/) for the server's CUDA version.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
python -m idsmultiplicity plan --config configs/unsw_nb15.yaml
```

The **real benchmark files are not included**. Obtain them from providers and place them in the expected folders described in [docs/DATASETS.md](docs/DATASETS.md). The loader does **not** scrape or mirror research datasets automatically.

## First run: UNSW-NB15

Place official `UNSW_NB15_training-set.csv` (175,341 records) and `UNSW_NB15_testing-set.csv` (82,332 records) into `data/raw/unsw_nb15/`, without renaming them.

```bash
# First check your experimental design without training.
idsmultiplicity plan --config configs/unsw_nb15.yaml

# Train or resume origin 1, with 63 neural candidates.
idsmultiplicity train --config configs/unsw_nb15.yaml --origin 1

# A genuinely selective Rashomon threshold (absolute validation accuracy).
idsmultiplicity disagree --run-dir runs/unsw_nb15/origin_seed1 --tolerance 0.001
idsmultiplicity diagnose --run-dir runs/unsw_nb15/origin_seed1

# SHAP compares fixed common test instances and the same class-1 logit.
idsmultiplicity explain --run-dir runs/unsw_nb15/origin_seed1 --tolerance 0.001 --background 64 --samples 128
idsmultiplicity plot --run-dir runs/unsw_nb15/origin_seed1 --tolerance 0.001

# Re-run all ten origins, optionally including SHAP (computationally expensive).
idsmultiplicity run-all --config configs/unsw_nb15.yaml --tolerance 0.001
# Add --with-shap if desired.
```

**To do a low-cost smoke run on the actual dataset** (five candidates, two training epochs), use `configs/smoke.yaml` and `--origin 1`; the unit tests additionally create their own synthetic CSV fixture and verify the whole train → disagreement → SHAP pipeline.

## Dataset support

| Config | Source and loader | Protocol | Status |
|---|---|---|---|
| `nsl_kdd.yaml` | NSL-KDD KDDTrain+/KDDTest+ files | Official split | Synthetic-schema test; original data not bundled |
| `unsw_nb15.yaml` | UNSW-NB15 official train/test CSVs | Official split | End-to-end synthetic test |
| `nf_*_v3.yaml` (four) | UQ NetFlow-v3 CSVs | Reproducible random test split | Synthetic-schema test; recommend explicit temporal split if dates exist |
| `ciciot2023.yaml` | CICIoT2023 extracted CSVs | Random test split | Synthetic-schema test; flow/session leakage risk |
| `ciciomt2024.yaml` | CICIoMT2024 Wi-Fi/MQTT train/test CSV trees | Official split | Synthetic-schema test; **not** compatible with raw Bluetooth PCAPs |
| `cse_cic_ids2018.yaml` | CSE-CIC-IDS2018 CICFlowMeter CSVs | Temporal test and validation splits | Synthetic-schema test; date parsing and label integrity require review |
| `bccc_cse_cic_ids2018.yaml` | BCCC-CSE-CIC-IDS2018 CSVs | Random *exploratory* split | Schema provisional; requires provider access and audit |

> The four NetFlow-v3 datasets are published with comparable 53-feature schemas, but old traffic underlies several new releases. Do not interpret “v3” as contemporary attack capture dates. Very large datasets are capped to a **reproducible uniform sample per source file** by the supplied profile. The resulting study evaluates a sampled benchmark, not the full provider distribution; disclose the caps in the paper.

## Experimental design

Same **PyTorch MLP** as in the predecessor: linear layer → activation → BatchNorm → dropout for each hidden layer, two-class linear output; cross-entropy, AdamW (LR 1e-3, weight decay 1e-4), 20 epochs maximum, validation-loss early stopping with patience 5. Default baseline architecture `[128, 64]` ReLU, dropout 0.2.

- 1 baseline model, 20 different initialization seeds;
- 20 bootstrap-resampled model fits;
- 20 different width/activation combinations (7 widths × 3 activations, excluding the baseline);
- 2 smaller stratified training fractions (75%, 50%).

The 10 origins use disjoint RNG namespaces for training, data resampling and (where applicable) validation splits. **The external test split is held constant** across origins, and data preprocessing is fit on the training partition only. For standard random-split datasets, the test seed is fixed while the train/validation seed changes. Test data are never used to select Rashomon members.

Default `ε=0.001` is an **absolute validation-accuracy tolerance**, not one percent; a sensitivity run can pass `--tolerance 0.005` separately. For cross-dataset comparisons, report the same tolerance policy, selected set size, confidence intervals over origin runs and relative vs absolute differences. See [docs/METHODS.md](docs/METHODS.md).

## What is saved

A run at `runs/<dataset>/origin_seed<N>/` produces:

```text
config.resolved.json       train/config provenance
split_manifest.json        protocol, feature schema, source file list, raw attack labels
prepared_data.npz          transformed train/validation/test arrays
preprocessor.joblib        fit-on-training sklearn preprocessing
metadata_{train,val,test}.csv  source identity and attack categories
training_plan.csv          exact architecture/seed/fraction per candidate
models/*.pt                PyTorch checkpoints (safe tensor-only loading)
model_results/*.json       cached metrics for checkpoint resume
predictions/*.npz          held-out predictions, keyed by model ID
metrics.csv                validation and test metrics
analysis/eps_0p00100/
  rashomon_models.csv      validation-selected models
  disagreement_summary.csv all unordered prediction pairs
  sample_disagreement.csv  consensus, conflict, failures, attack labels
  security_by_attack_type.csv
  multiplicity_summary.json
  shap/                    complete per-sample, per-feature attack-logit attributions
  figures/*.pdf            publication-ready figures
 diagnostics/              split prevalence, unseen attack types, KS shift
```

Use `idsmultiplicity aggregate --run-dirs runs/unsw_nb15/origin_seed{1..10} --tolerance 0.001 --out-dir reports/unsw_nb15` for origin-level statistics and cross-origin persistence **only when sample identities match**. All generated data, models, and raw traffic are excluded by `.gitignore`; publish only vetted aggregated results when license/privacy policies permit.

## Reproducibility and scientific constraints

1. Split train/test **before fitting preprocessing**, use the same feature ordering for all models, and record exact seed namespaces. Avoid `attack_cat`, `Label`, IDs, or attack-category-derived fields as predictors.
2. Random flow splits can leak nearly identical connections and capture-session information. The official or temporal split should be preferred when defensible; a random split is exploratory for many IDS collections.
3. The source files must be checked for train/test overlap, label consistency, missing values, duplicates, class prevalence, timestamp format, and attack scenario boundaries before published comparisons. The repo does not guarantee that these provider datasets satisfy an independent-sample assumption.
4. SHAP attribution depends on the output function, background and sample selection; comparisons within a run use the **same class-1 output logit**, background and test instances. Global importance cosine uses the **full feature vector**, not top-50 truncation.
5. The baseline does not claim that all 63 models comprise the theoretical Rashomon set. Means and SDs use origins as replication units, **not the 1,953 dependent model pairs**.
6. `.npz`/`.pt` checkpoint artifacts are derived from data and should not be redistributed without checking provider terms; no model weights or original CSVs are provided in this archive.

## Paper material

The `paper/proposal/` directory contains the previous ICMCIS-2027 manuscript draft (LaTeX and figures) to make the repository self-contained as a *working paper*. The figures currently reflect the NSL-KDD experiments from the prior work. **The new dataset experiments do not populate that manuscript automatically**; update claims/figures and recompile after actual new runs.

## Research code origin & release

The factorized training design was adapted from the author's existing project; differences (sampling, split handling, fully dimensional SHAP, file formats) are described in [docs/METHODS.md](docs/METHODS.md). This archive intentionally includes **no LICENSE** yet: select a software license and verify permission to publish work produced within institutional/employer research before placing it on GitHub. See [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md).

## Starting a **new** GitHub repository

The archive is not an already-created GitHub repository. After extraction and review, from inside its root:

```bash
git init
git add .
git commit -m "Initial IDS model-multiplicity research pipeline"
# Requires GitHub CLI and a user-approved new repository name/visibility:
gh repo create intrusion-multiplicity-benchmark --private --source=. --remote=origin --push
```

Choose a software license and complete the release checklist **before changing visibility to public**. The old repository is not modified by these instructions.
