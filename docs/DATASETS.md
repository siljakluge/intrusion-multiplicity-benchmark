# Dataset setup and provenance

All downloads are manual from the original providers. **Do not commit any original data** to the public GitHub repository. The profiles include heuristically matched column names; validate each provider archive before running a full benchmark. Preprocessing automatically drops known label/identifier columns, infers categorical columns, and fits only on training data. It refuses train/test file overlap and incompatible CSV column sets.

| Profile | Root relative to repo | Expected files | Provider | Notes |
|---|---|---|---|---|
| NSL-KDD | `data/raw/nsl_kdd/` | `KDDTrain+.txt`, `KDDTest+.txt` | [NSL-KDD reference / mirror provenance](https://github.com/defcom17/NSL_KDD) | 43 columns without header (41 predictors, label, difficulty); official split; binary normal vs attack. |
| UNSW-NB15 | `data/raw/unsw_nb15/` | `UNSW_NB15_training-set.csv`, `UNSW_NB15_testing-set.csv` | [UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Official 175,341/82,332 split; target `label`; subtype `attack_cat`; categories `proto`, `service`, `state`. **Academic-use grant, commercial use separate.** |
| NF-UNSW-NB15-v3 | `data/raw/nf_unsw_nb15_v3/` | extracted `*.csv` under root | [UQ NetFlow v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Target `Label`, subtype `Attack` (inspect real column spelling). |
| NF-ToN-IoT-v3 | `data/raw/nf_ton_iot_v3/` | extracted `*.csv` | [UQ NetFlow v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Very large; sample cap applied. |
| NF-BoT-IoT-v3 | `data/raw/nf_bot_iot_v3/` | extracted `*.csv` | [UQ NetFlow v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Extreme attack/benign imbalance. |
| NF-CSE-CIC-IDS2018-v3 | `data/raw/nf_cse_cic_ids2018_v3/` | extracted `*.csv` | [UQ NetFlow v3](https://staff.itee.uq.edu.au/marius/NIDS_datasets/) | Very large; temporal indicators may require provider-specific split configuration. |
| CICIoT2023 | `data/raw/ciciot2023/` | extracted CSV directory; matched recursively | [UNB CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | CSV, not PCAP; 33 attacks. Random split of captured flows may be optimistic; sampling by source file shifts class priors. |
| CICIoMT2024 | `data/raw/ciciomt2024/` | Wi-Fi/MQTT extracted CSV under `**/train/**/*.csv` and `**/test/**/*.csv` | [UNB CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html) | CSV is provided for Wi-Fi/MQTT; Bluetooth test/train are PCAP, outside this loader. |
| CSE-CIC-IDS2018 | `data/raw/cse_cic_ids2018/` | CICFlowMeter daily `*.csv` | [UNB CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | `Timestamp` sorted for temporal split. Provider CSVs may contain NaN/inf or inconsistent extraction. Verify capture-day and victim leakage. |
| BCCC-CSE-CIC-IDS2018 | `data/raw/bccc_cse_cic_ids2018/` | provider CSVs | [York BCCC](https://www.yorku.ca/research/bccc/ucs-technical/cybersecurity-datasets-cds/large-scale-intrusion-detection-dataset-bccc-cse-cic-ids2018/) | Request-only release, ~46M observations/300 features. The default random sample/split is exploratory, not yet paper-grade. Inspect exact schema and split by scenario before presenting results. |

## Critical protocol details

* **Labels:** Dataset-specific profiles select a `Label`/`label` column, decide whether `0` means benign or whether a textual value means benign, and retain any available attack subtype as evaluation metadata. Review actual label values before training; unknown *textual* labels currently map to attack. An incorrect benign alias can invalidate the study.
* **Sampling:** `max_rows_per_file` performs reproducible random reservoir sampling within each CSV. It bounds RAM but changes the measured population, can under-represent rare attacks or bias source-file proportions. Report sample sizes and selection mechanism. To use full files, set `max_rows_per_file: null` and ensure sufficient disk/RAM.
* **Splits:** Official train/test archives are preferred. For other data, `split: temporal` requires a parsable time column; `split: group` requires a correctly specified stable scenario/session field. Random flow splits can violate deployment independence; use with caution.
* **Train-test isolation:** `test_split_seed` and `sampling_seed` are fixed across origins. The per-origin `split_seed` changes the validation split within the training pool. Temporal/group split arrangements should be reviewed for class coverage and no overlap.
* **Timestamps and network identifiers:** `Timestamp`, IP/MAC addresses, flow IDs and attack labels are excluded as predictors by default; some port fields remain intentionally in the feature space. If a label proxy survives under an unrecognized name, add it to `drop_columns`.
* **Separate protocol subsets:** For CICIoMT2024, prefer Wi-Fi/MQTT extracted features and keep BLE PCAP experiments separate. `read_files` intentionally fails on mixed, incompatible feature schemas instead of silently aligning unknown columns.
* **Data license:** Dataset access and use are not necessarily the same as the right to redistribute a mirror or derived examples. Store provider files outside Git and comply with citation and licensing terms.

## Custom CSV benchmark

Create `configs/my_dataset.yaml` by copying `unsw_nb15.yaml` and updating `dataset.root`, `train_files`/`test_files` or `files`/`split`, `label_candidates` or `label_column`, `binary_label`, `attack_type_candidates`, `categorical_columns`, and `drop_columns`. The generic CSV code requires just one binary target column and at least one numeric or categorical feature. A temporary, schema-specific test should be written before launching all origins.
