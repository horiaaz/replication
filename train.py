"""Train EMS-YOLO.

Usage:
    python train.py --config configs/ems_resnet10_t1.yaml \
        --data-root /path/to/coco --output runs/t1
    python train.py --config configs/ems_resnet10_t3.yaml \
        --data-root /path/to/coco --output runs/t3 \
        --pretrained runs/t1/best.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import CocoYOLODataset, yolo_collate
from models import build_model
from utils import ComputeLoss, decode_predictions, evaluate_coco, predictions_to_coco_json


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_pretrained(model: nn.Module, ckpt_path: str) -> None:
    """Load weights from a checkpoint with possibly different T."""
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if "model" in sd else sd
    # Strip module prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[pretrained] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing:", missing[:5], "..." if len(missing) > 5 else "")
    if unexpected:
        print("  unexpected:", unexpected[:5], "..." if len(unexpected) > 5 else "")


def cosine_lr(epoch: int, total_epochs: int, lr0: float, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return lr0 * (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return lr0 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, val_loader, cfg, output_dir, coco_gt_path, idx_to_id, device):
    model.eval()
    all_preds = []
    for imgs, _labels, meta in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        results = decode_predictions(preds, model.strides, model.nc, model.na,
                                     conf_thresh=0.001)
        all_preds.extend(
            predictions_to_coco_json(results, meta, idx_to_id, cfg["data"]["img_size"])
        )
    if not all_preds:
        return {"mAP@0.5:0.95": 0.0, "mAP@0.5": 0.0}
    pred_path = Path(output_dir) / "preds.json"
    with open(pred_path, "w") as f:
        json.dump(all_preds, f)
    return evaluate_coco(coco_gt_path, all_preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", required=True, help="COCO root with annotations/ and images/")
    ap.add_argument("--output", required=True)
    ap.add_argument("--pretrained", default=None, help="Optional checkpoint to init from")
    ap.add_argument("--resume", default=None, help="Resume training from checkpoint")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------- Data ---------
    train_ds = CocoYOLODataset(args.data_root, "train2017",
                               img_size=cfg["data"]["img_size"], augment=True)
    val_ds = CocoYOLODataset(args.data_root, "val2017",
                             img_size=cfg["data"]["img_size"], augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=yolo_collate,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=yolo_collate,
        pin_memory=True,
    )
    coco_gt_path = str(Path(args.data_root) / "annotations" / "instances_val2017.json")

    # --------- Model ---------
    model = build_model(cfg).to(device)
    if args.pretrained:
        load_pretrained(model, args.pretrained)
    # Make sure T matches the config even if loaded from a different-T checkpoint
    model.set_T(cfg["model"]["T"])

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] EMS-YOLO T={model.T}  params={n_params/1e6:.2f}M")

    # --------- Loss + Optim ---------
    loss_fn = ComputeLoss(model)
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg["train"]["lr0"],
        momentum=cfg["train"]["momentum"],
        weight_decay=cfg["train"]["weight_decay"],
        nesterov=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    start_epoch = 0
    best_map = 0.0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        optim.load_state_dict(ck["optim"])
        start_epoch = ck["epoch"] + 1
        best_map = ck.get("best_map", 0.0)
        print(f"[resume] from epoch {start_epoch}, best mAP {best_map:.3f}")

    epochs = cfg["train"]["epochs"]
    accum = cfg["train"]["accum_steps"]

    # --------- Training loop ---------
    for epoch in range(start_epoch, epochs):
        model.train()
        lr = cosine_lr(epoch, epochs, cfg["train"]["lr0"], cfg["train"]["warmup_epochs"])
        for pg in optim.param_groups:
            pg["lr"] = lr

        t0 = time.time()
        running = {"box": 0.0, "obj": 0.0, "cls": 0.0, "total": 0.0}
        optim.zero_grad(set_to_none=True)

        for step, (imgs, labels, _meta) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"]):
                preds = model(imgs)
                loss, parts = loss_fn(preds, labels)
                loss = loss / accum

            scaler.scale(loss).backward()
            if (step + 1) % accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            for k, v in parts.items():
                running[k] += float(v)

            if (step + 1) % 100 == 0:
                msg = " ".join(f"{k}={v/(step+1):.4f}" for k, v in running.items())
                print(f"[epoch {epoch} step {step+1}/{len(train_loader)} lr={lr:.4g}] {msg}")

        dt = time.time() - t0
        print(f"[epoch {epoch}] done in {dt/60:.1f} min")

        # --------- Eval ---------
        if (epoch + 1) % cfg["train"]["eval_every"] == 0 or epoch == epochs - 1:
            metrics = evaluate(model, val_loader, cfg, out, coco_gt_path,
                               val_ds.idx_to_id, device)
            print(f"[epoch {epoch}] mAP@0.5={metrics['mAP@0.5']:.4f} "
                  f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f}")

            # Save best
            if metrics["mAP@0.5"] > best_map:
                best_map = metrics["mAP@0.5"]
                torch.save(
                    {"model": model.state_dict(), "optim": optim.state_dict(),
                     "epoch": epoch, "best_map": best_map, "cfg": cfg},
                    out / "best.pt",
                )
                with open(out / "best_metrics.json", "w") as f:
                    json.dump(metrics, f, indent=2)

        # --------- Periodic save ---------
        if (epoch + 1) % cfg["train"]["save_every"] == 0:
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(),
                 "epoch": epoch, "best_map": best_map, "cfg": cfg},
                out / "last.pt",
            )


if __name__ == "__main__":
    main()
