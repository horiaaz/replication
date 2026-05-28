"""Evaluate an EMS-YOLO checkpoint on COCO val2017.

Usage:
    python evaluate.py --config configs/ems_resnet10_t5.yaml \
        --ckpt runs/t5/best.pt --data-root /path/to/coco
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from data import CocoYOLODataset, yolo_collate
from models import build_model
from utils import decode_predictions, evaluate_coco, predictions_to_coco_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--output", default="eval_out")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_ds = CocoYOLODataset(args.data_root, "val2017",
                             img_size=cfg["data"]["img_size"], augment=False)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"],
                            collate_fn=yolo_collate, pin_memory=True)

    model = build_model(cfg).to(device)
    sd = torch.load(args.ckpt, map_location="cpu")
    state = sd["model"] if "model" in sd else sd
    model.load_state_dict(state)
    model.eval()
    print(f"[model] loaded {args.ckpt}  T={model.T}")

    coco_gt_path = str(Path(args.data_root) / "annotations" / "instances_val2017.json")
    all_preds = []
    with torch.no_grad():
        for imgs, _labels, meta in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            preds = model(imgs)
            results = decode_predictions(preds, model.strides, model.nc, model.na,
                                         conf_thresh=0.001)
            all_preds.extend(
                predictions_to_coco_json(results, meta, val_ds.idx_to_id, cfg["data"]["img_size"])
            )

    if not all_preds:
        print("[eval] no predictions")
        return
    with open(out / "preds.json", "w") as f:
        json.dump(all_preds, f)
    metrics = evaluate_coco(coco_gt_path, all_preds)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
