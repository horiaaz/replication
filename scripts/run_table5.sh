#!/usr/bin/env bash
# Reproduce Table 5: T=1 then fine-tune T=3, T=5, T=7 from the T=1 checkpoint.
set -euo pipefail

COCO_ROOT=${COCO_ROOT:?"set COCO_ROOT to your COCO 2017 directory"}
OUT=${OUT:-runs}

mkdir -p "$OUT"

echo "=========================================="
echo "[Table 5] Step 1/4 — train T=1 from scratch"
echo "=========================================="
python train.py --config configs/ems_resnet10_t1.yaml \
    --data-root "$COCO_ROOT" --output "$OUT/t1"

T1_CKPT="$OUT/t1/best.pt"

for T in 3 5 7; do
    echo "=========================================="
    echo "[Table 5] Fine-tune T=$T from T=1 ckpt"
    echo "=========================================="
    python train.py --config "configs/ems_resnet10_t${T}.yaml" \
        --data-root "$COCO_ROOT" --output "$OUT/t${T}" \
        --pretrained "$T1_CKPT"
done

echo
echo "==================================================================="
echo "[Table 5] Final results"
echo "==================================================================="
for T in 1 3 5 7; do
    if [[ -f "$OUT/t${T}/best_metrics.json" ]]; then
        echo "T=$T :"
        cat "$OUT/t${T}/best_metrics.json"
        echo
    fi
done
