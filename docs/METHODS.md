# Methods and reproducibility specification

## Candidate population (factorized design)

For every origin `o ∈ {1,...,10}`, generate MLP candidates in five blocks: baseline (1), 20 independently initialized MLPs, 20 bootstrap-resampled training sets, 20 architecture/activation combinations, and two 75%/50% stratified subsets. The block sizes reproduce the factorized NSL-KDD experimental plan, not a complete factorial or causal design. The baseline network has widths `[128,64]`, ReLU, BatchNorm, dropout 0.2, AdamW learning rate `1e-3` and weight decay `1e-4`, cross-entropy, batch size 256 and validation-loss early stopping (max 20 epochs, patience five). MLP checkpoints and exact seed schedules are saved per origin. No XGBoost models are included by default.

Some details deliberately improve on the previous code: validation cross-entropy is weighted by batch size (rather than a mean of batch means), raw source sample identifiers are preserved, SHAP global comparisons use **full** attribution vectors instead of top-50-truncated vectors, cross-origin test holdouts stay fixed when using a random split, and smaller Rashomon thresholds are available without retraining. These details make the new experiment **methodologically related but not byte-for-byte identical** to prior saved NSL-KDD models. Old reported numerical results should not be presented as reruns of this new repository.

## Split and transform

Use a provider's official train/test split when available. Derive the validation split from the training source only. If official splits are absent, set a stable `test_split_seed` and vary per-origin validation seeds; use an explicit temporal or group split when source fields allow it. Any per-file sampling is deterministic and happens before splitting; for huge CSV files it uses the smallest random-priority reservoir without constructing a full DataFrame in memory. All missing-value imputers, numeric scalers, categorical vocabularies, and one-hot encoders fit on **train only**. Preserve the same feature order for validation and test. Raw attack-family labels and provenance are metadata, never predictors.

## Rashomon selection

For origin `o`, candidate pool `F_o`, validation score `v_o(f)` and absolute tolerance ε:

`R_o(ε) = {f in F_o : v_o(f) >= max_{g in F_o} v_o(g) - ε}`.

Selection uses validation accuracy or validation macro-F1, not held-out test accuracy. `ε=0.001` corresponds to **0.1 percentage points of accuracy**. This is a finite empirical Rashomon set; it is not an exhaustive enumeration of a near-optimal hypothesis class.

## Predictive multiplicity

For a fixed held-out test population of size N, the pairwise disagreement is `D(f,g) = (1/N) Σ_x I[y_hat_f(x) != y_hat_g(x)]`. Sample ambiguity is the fraction of test instances for which at least two selected models predict different binary labels. The pointwise conflict ratio is `c(x) = 1 - max_y Σ_f I[y_hat_f(x) = y] / |R_o|`. Vote entropy is computed from the relative attack and normal votes, with 0 for unanimity. Consensus states are `unanimous_normal`, `unanimous_attack`, and `contested`. An `unanimous_false_negative` is an attack test instance predicted normal by all selected models; it is not equivalent to the false-negative rate of an individual model. If `|R_o| = 1`, pairwise disagreement is undefined and the software reports 0 solely for convenience (consult selected set size before interpretation).

## Explanation multiplicity

Compute `shap.DeepExplainer` on the **attack-class (index 1) logit**, with a common random background sample (default 64 training records) and the same test instances (default 128) for every selected model. The default explanation sample balances contested and unanimous events where possible; consequently, **unweighted SHAP summary statistics are not estimates for the natural test distribution**. Global importance per feature is the mean absolute attack-logit attribution across these explained instances. Between model pairs, calculate cosine distance of full global importance vectors and Jaccard similarity of the top-k most influential feature sets. For local variation, compute the across-model population variance (`ddof=0`) and range of each sample/feature attribution, as well as sign instability `min(p_positive, 1-p_positive)`. Aggregate local per-sample quantities by averaging across features. Report differences by prediction consensus and Spearman association with sample-level conflict. This is an observational association, not evidence of causality or calibrated uncertainty.

Important: fixed background instances, model outputs and SHAP explainer choices are essential for comparability. Across datasets with different numbers of features or output distributions, absolute SHAP magnitudes and raw variances are **not directly comparable**. To make statements across datasets, prefer shared measures such as rank overlap, categorical consensus, performance gap, and normalized within-dataset quantities.

## Security error audit

Subtype metadata are preserved (e.g., UNSW `attack_cat`, NetFlow `Attack`). Summarize the frequency of contested predictions, unanimous missed attacks and unanimous false positives by subtype. `unseen_attack_type` is defined as a test attack whose **exact normalized subtype label** does not appear in the origin's training partition. This only makes sense when source labels share the same taxonomy; it can be misleading when training and test use different label granularity, spelling or unknown/aggregate classes. Attack-type counts in the test split are the denominators for security-failure proportions; rare subtypes need intervals or careful caveats.

## Aggregation and independent units

Compute all pairwise and sample-level statistics separately by origin. When comparing multiple origins, report mean and sample SD across origins (n=10), not across model pairs. All origins may share the same held-out samples. Cross-origin persistence calculations are permitted only when `source_file`, `source_row` and labels agree exactly in the same order. The sample index is local to each run and cannot substitute for a verified common test population.

## Deviations from NSL-KDD paper proposal

The prior ICMCIS draft used NSL-KDD with 630 MLP candidates and classwise SHAP comparisons. This repository uses one class (attack) for its explicit cross-model explanation audit, a potentially balanced conflict-focused sample, and fully dimensional SHAP vectors. Thus *do not copy NSL-KDD numbers from the previous manuscript into a table claiming they came from this repository*. Retrain NSL-KDD if numerical parity and the same code base are necessary, or clearly describe the old analyses as a prior pipeline.

## External validity / release

Temporal ordering, leakage, duplicate flows, attack taxonomy, class prevalence, per-file sampling and true scenario independence are dataset-specific and require documented human review. The generic profiles are an extensible starting point; they do not constitute verified benchmarks merely because their unit tests pass with synthetic schemas.
