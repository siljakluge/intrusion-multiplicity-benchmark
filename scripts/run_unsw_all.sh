#!/usr/bin/env bash
set -euo pipefail
idsmultiplicity plan --config configs/unsw_nb15.yaml --origin 1
idsmultiplicity run-all --config configs/unsw_nb15.yaml --tolerance 0.001
for seed in {1..10}; do
  idsmultiplicity explain --run-dir "runs/unsw_nb15/origin_seed${seed}" --tolerance 0.001 --background 64 --samples 128
  idsmultiplicity plot --run-dir "runs/unsw_nb15/origin_seed${seed}" --tolerance 0.001
done
idsmultiplicity aggregate --run-dirs runs/unsw_nb15/origin_seed{1..10} --tolerance 0.001 --out-dir reports/unsw_nb15
