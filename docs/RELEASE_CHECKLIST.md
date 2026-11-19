# Checklist before public paper/code release

- [ ] Confirm permission from employer/institution to release this code and any code derived from prior employment research.
- [ ] Choose and add a software `LICENSE` file (MIT, BSD-3-Clause, Apache-2.0, etc.) after verifying that you have redistribution rights for predecessor code; the archive intentionally contains no license.
- [ ] Check official dataset terms; do **not** publish raw PCAPs, provider CSVs or restricted dataset mirrors.
- [ ] Inspect actual schemas and correct the dataset-specific loaders where necessary. Synthetic fixture tests are insufficient to validate real provider files.
- [ ] Prefer official, temporal or group-independent test sets. Evaluate and document file/session/flow duplicates and attack-family taxonomies.
- [ ] Record exact dataset version, provider URL, download date, raw-file checksums, dataset license, sampling cap, class prevalence, split identifiers, and any exclusion filters.
- [ ] Freeze runtime versions: `python -m pip freeze > environment-lock.txt`; record Python, CUDA, GPU, torch, SHAP, NumPy and sklearn versions.
- [ ] Run `pytest -q`, one local smoke test, then 10 complete origins of the selected primary and comparison datasets.
- [ ] Run sensitivity analyses with multiple Rashomon tolerances and validation metrics; never choose a threshold using held-out test results without disclosing it as post hoc.
- [ ] Verify all test-origin sample identities before persistence analyses and use origin-level rather than pair-level uncertainty intervals.
- [ ] Check source-wise sampling/imbalance, subgroup denominators, and failures among unanimously missed events.
- [ ] Decide whether to include saved model checkpoints/SHAP arrays. They may be large or fall under the dataset provider's derivative-data terms.
- [ ] Replace paper/proposal's historical figures with outputs from the release run and update all numerical claims, citations, and caveats accordingly.
- [ ] Add GitHub release tag, full commit SHA, dataset checksums, and artifact manifest to the final paper/reproducibility statement.
- [ ] Verify the latest official ICMCIS 2027 page limit, IEEE template, conference text, and review requirements.
