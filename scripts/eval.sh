#!/usr/bin/env bash
# Evaluate all four Table 5 checkpoints.
set -euo pipefail
COCO_ROOT=${COCO_ROOT:?"set COCO_ROOT"}
OUT=${OUT:-runs}
for T in 1 3 5 7; do
    if [[ -f "$OUT/t${T}/best.pt" ]]; then
        echo "==== T=$T ===="
        python3 evaluate.py --config "configs/ems_resnet10_t${T}.yaml" \
            --ckpt "$OUT/t${T}/best.pt" \
            --data-root "$COCO_ROOT" \
            --output "$OUT/t${T}/eval"
    fi
done
