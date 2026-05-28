# Python File Overview

## Entry points

### `train.py`
Main training script. Loads a YAML config, builds the dataset and model, runs the training loop with SGD + cosine LR + AMP (mixed precision), and saves `best.pt` and `last.pt` checkpoints. Evaluates on COCO val at the interval set by `eval_every`. Supports resuming from a checkpoint (`--resume`) and initialising from a different-T checkpoint (`--pretrained`).

### `evaluate.py`
Standalone evaluation script. Loads a saved checkpoint, runs inference on COCO val2017, and prints mAP@0.5 and mAP@0.5:0.95. Use this after training to score a specific checkpoint without re-running the full training loop.

---

## `models/`

### `models/neurons.py`
Defines the core spiking neuron primitives:
- **`RectSurrogate`** — rectangular surrogate gradient (enables backprop through the non-differentiable spike threshold).
- **`LIF`** — Leaky Integrate-and-Fire neuron. Runs a step-by-step loop over the time dimension `T`, maintaining membrane potential and emitting spikes. Also has `forward_last_v` which returns the final membrane potential instead of the spike train (used by the detector head).
- **`TDBN`** — Threshold-dependent BatchNorm. Standard BN2d applied across the `(T, B)` axes, with its gamma initialised to `alpha * V_th` as described in the paper.

### `models/ems_blocks.py`
Implements the EMS residual building blocks:
- **`TConv2d` / `TMaxPool2d`** — thin wrappers that fold the time dimension into the batch axis so standard `nn.Conv2d` / `nn.MaxPool2d` can operate on `(T, B, C, H, W)` tensors.
- **`LCB`** (LIF → Conv → BN) — the basic "spike conv" unit used throughout the network.
- **`EMSBlock1`** — residual block for constant or decreasing channel counts. Identity shortcut (optionally with a maxpool + 1×1 spike conv if spatial or channel dimensions change).
- **`EMSBlock2`** — residual block for increasing channel counts. Shortcut uses a concat of the maxpooled input and a 1×1 projection to fill the extra channels cheaply (the "feature reuse" trick from Sec. 4.2).
- **`MSBlock`** — simple residual block (no shortcut transform) used inside the detection head.

### `models/ems_resnet.py`
Defines the **EMS-ResNet10** backbone. A stem conv → 4 stages (each one block) producing feature maps at stride/16 (P4) and stride/32 (P5), which are fed to the detector head. Channel widths default to `[64, 128, 256, 512]` for the COCO setup.

### `models/detector.py`
Defines the **spiking detection head**:
- **`MembraneHead`** — reads out the LIF membrane potential (not spikes) at the final step via `forward_last_v`, then applies a 1×1 conv to produce the raw `(B, A*(5+nc), H, W)` prediction grid. This is the one unavoidably dense (MAC) operation.
- **`EMSYOLOHead`** — 2-scale head (P4 and P5). Processes each scale through MS-Blocks, fuses them top-down (upsample P5 → concat with P4), and feeds each into a `MembraneHead`.

### `models/yolo.py`
Wraps backbone + head into the full **`EMSYOLO`** model. The `lift` method repeats the input image `T` times along a new leading time axis (the standard repeat-input encoding). `set_T` lets you swap the timestep count after loading a checkpoint (used by the T=1 → T=3/5/7 fine-tuning recipe). `build_model(cfg)` constructs the model from a YAML config dict.

---

## `data/`

### `data/coco.py`
- **`CocoYOLODataset`** — PyTorch `Dataset` for COCO 2017. Parses the annotation JSON, maps COCO's non-contiguous category IDs to contiguous 0..79 indices, applies letterbox resizing, and returns images as `(3, H, W)` float tensors alongside YOLO-format labels `(N, 5)` in `[cls, cx, cy, w, h]` normalised to `[0, 1]`.
- **`yolo_collate`** — custom collate function that stacks images into `(B, 3, H, W)` and concatenates all labels into a single `(M, 6)` tensor where column 0 is the batch index.

### `data/transforms.py`
Image preprocessing utilities:
- **`letterbox`** — resizes an image to a square canvas while preserving aspect ratio (padding with grey), and updates the YOLO label coordinates accordingly.
- **`random_hflip`** — random horizontal flip with probability `p`, updating label `cx` coordinates.
- Contains a comment stub showing how to plug in Mosaic augmentation from ultralytics (not implemented here).

---

## `utils/`

### `utils/loss.py`
- **`bbox_ciou`** — computes Complete IoU between predicted and target boxes in `(cx, cy, w, h)` format.
- **`ComputeLoss`** — callable loss class. For each prediction scale it matches targets to anchors, decodes predicted boxes, and computes CIoU box loss + BCE objectness loss (with IoU-as-target for positives) + BCE classification loss. Returns a scalar total loss and a dict of components.

### `utils/anchors.py`
- **`build_anchors`** — returns the default YOLOv3-tiny anchor boxes (in pixels) for strides 16 and 32.
- **`match_targets`** — assigns each ground-truth box to the best-matching anchor(s) on a given scale using width/height ratio thresholding (`anchor_thresh=4.0`), returning the class indices, target boxes, grid indices, and anchor sizes needed by the loss.

### `utils/nms.py`
- **`decode_predictions`** — decodes raw prediction grids into `(x1, y1, x2, y2, score, cls)` boxes, applies a confidence threshold, and runs batched NMS (via `torchvision.ops.batched_nms`). Returns one `(N, 6)` tensor per image.

### `utils/coco_eval.py`
- **`predictions_to_coco_json`** — converts the model's per-image box tensors back to COCO submission format, undoing the letterbox padding and rescaling boxes to the original image coordinates.
- **`evaluate_coco`** — wraps `pycocotools` to compute and return mAP@0.5:0.95 and mAP@0.5.
