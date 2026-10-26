#!/usr/bin/env bash
# End-to-end check on a synthetic MMRDR-shaped dataset (CPU, about 10 minutes), in the same order
# as run_all.sh. Verifies that every script runs; the numbers produced are meaningless.
set -euo pipefail
CONFIG=configs/smoke_test.yaml

python scripts/make_synthetic_mmrdr.py --out data/synthetic
python scripts/prepare_data.py   --config $CONFIG --overwrite
python scripts/linear_probe.py   --config $CONFIG
python scripts/train.py          --config $CONFIG
python scripts/train.py          --config $CONFIG --set experiment_name=smoke_resnet18_unweighted train.class_weighting=none
python scripts/select_model.py   --config $CONFIG
python scripts/train_seeds.py    --config $CONFIG --seeds 1 2
python scripts/select_model.py   --config $CONFIG   # replicates must not be selected
python scripts/evaluate.py       --config $CONFIG
python scripts/explain.py        --config $CONFIG
python scripts/robustness.py     --config $CONFIG
python scripts/prepare_idrid.py  --src "data/synthetic/idrid_like/A. Segmentation" --out data/synthetic_idrid
python scripts/localisation_external.py --config $CONFIG --images data/synthetic_idrid/images \
       --masks data/synthetic_idrid/masks --lesions MA HE EX SE --name idrid
python scripts/collect_results.py --config $CONFIG
echo "Smoke test finished: see outputs_smoke/report/tables.md"
