# EMS-YOLO — Table 5 Replication Skeleton

Standalone PyTorch skeleton to reproduce **Table 5** of Su et al., *Deep
Directly-Trained Spiking Neural Networks for Object Detection* (ICCV 2023).

Table 5 ablates the number of LIF timesteps `T ∈ {1, 3, 5, 7}` for
**EMS-ResNet10** as backbone on **COCO 2017**, with a 2-head YOLOv3-tiny-style
detector. Target numbers:

| T | mAP@0.5 | mAP@0.5:0.95 |
|---|---------|--------------|
| 1 | 0.328   | 0.162        |
| 3 | 0.362   | 0.184        |
| 5 | 0.367   | 0.189        |
| 7 | 0.383   | 0.199        |

The paper trains T=1 first and uses it as the init for T=3/5/7 to cut wall time.
The scripts in `scripts/` follow that recipe.

## What this is, and what it is not

This is a **clean-room skeleton**. It gives you:

- a faithful implementation of the LIF neuron, surrogate gradient, and tdBN
  (Sec. 3.1–3.2 of the paper),
- EMS-Block1 / EMS-Block2 (Fig. 1c, Fig. 2),
- EMS-ResNet10 backbone with the channel widths used in the paper for the
  COCO setup,
- a 2-head YOLOv3-tiny-style detector head whose multi-layer direct convs are
  replaced with EMS-Blocks (Sec. 4.3),
- a COCO `Dataset` + collate that returns YOLO-format labels,
- a training loop with SGD, lr=1e-2, Mosaic-style aug hook, AMP, gradient
  accumulation, multi-T config, and pretrained-from-T=1 loading,
- a pycocotools-based mAP evaluator.

It does **not** include:

- Mosaic augmentation in full (a hook is provided; plug ultralytics' or
  albumentations' implementation),
- the full YOLOv3 anchor-matching / loss bookkeeping at the polish of the
  ultralytics repo — a working CIoU-based loss is included that will train,
  but if you want bit-exact numbers you should swap in `ComputeLoss` from
  ultralytics/yolov3,
- pretrained ImageNet weights for EMS-ResNet10 (the paper trains from scratch
  with the T=1 → T>1 trick, which is what `scripts/run_table5.sh` does).

For a bit-exact reproduction, the safest route is still to fork
https://github.com/BICLab/EMS-YOLO and swap T in their config. This skeleton
is for understanding the moving parts and iterating on them.

## Layout

```
ems_yolo_table5/
├── models/
│   ├── neurons.py          # LIF + surrogate gradient + tdBN
│   ├── ems_blocks.py       # EMS-Block1, EMS-Block2, MS-Block, LCB
│   ├── ems_resnet.py       # EMS-ResNet10 backbone
│   ├── detector.py         # 2-head YOLOv3-tiny-style spiking head
│   └── yolo.py             # full EMS-YOLO model, wraps T-step inference
├── data/
│   ├── coco.py             # COCO Dataset + YOLO-format collate
│   └── transforms.py       # letterbox + (Mosaic hook)
├── utils/
│   ├── loss.py             # CIoU + obj + cls loss
│   ├── anchors.py          # anchor generation / matching
│   ├── nms.py              # batched NMS
│   └── coco_eval.py        # pycocotools mAP wrapper
├── configs/
│   └── ems_resnet10_t{1,3,5,7}.yaml
├── scripts/
│   ├── run_table5.sh       # full Table 5 sweep (T=1, then fine-tune to 3,5,7)
│   └── eval.sh
├── train.py
├── evaluate.py
└── requirements.txt
```

## Quickstart (single GPU)

```bash
pip install -r requirements.txt

# 1. Download COCO 2017 to /path/to/coco with the standard layout:
#    coco/{annotations,images/train2017,images/val2017}
export COCO_ROOT=/path/to/coco

# 2. Train T=1 from scratch
python train.py --config configs/ems_resnet10_t1.yaml \
    --data-root $COCO_ROOT --output runs/t1

# 3. Fine-tune T=3, T=5, T=7 from the T=1 checkpoint
python train.py --config configs/ems_resnet10_t3.yaml \
    --data-root $COCO_ROOT --output runs/t3 \
    --pretrained runs/t1/best.pt

python train.py --config configs/ems_resnet10_t5.yaml \
    --data-root $COCO_ROOT --output runs/t5 \
    --pretrained runs/t1/best.pt

python train.py --config configs/ems_resnet10_t7.yaml \
    --data-root $COCO_ROOT --output runs/t7 \
    --pretrained runs/t1/best.pt

# Or just run the whole sweep:
bash scripts/run_table5.sh
```

## Single-GPU notes

The paper uses 8× RTX 3090 with batch 32. On a single GPU you have two knobs:

1. **Gradient accumulation** — set `accum_steps` in the YAML so that
   `batch_size × accum_steps ≈ 32`. The lr schedule assumes effective batch 32.
2. **Image size** — paper uses 640. Drop to 416 if you're VRAM-bound; document
   the change since it will affect absolute mAP.

T scales VRAM and time roughly linearly because activations are replayed per
timestep. Rough estimates on a single 24 GB GPU at img 640, batch 8:

| T | VRAM (approx) | Time / epoch (approx) |
|---|---------------|-----------------------|
| 1 | 10 GB         | 1×                    |
| 3 | 16 GB         | 2.5×                  |
| 5 | 22 GB         | 4×                    |
| 7 | OOM at bs 8   | 5.5× (drop bs to 4)   |

300 epochs at T=1 on a single GPU is on the order of days, not hours. Realistic
plan: start at low resolution / fewer epochs to sanity-check the pipeline, then
scale up.
