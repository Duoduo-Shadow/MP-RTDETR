# MP-RTDETR

MP-RTDETR is a real-time detector for UAV aerial scenes with small and dense objects.  
This codebase keeps only the implementation-focused parts for reproducible training and evaluation.

Core modifications in this release:
- Feature/scale refinement blocks (`FSRA`)
- Multi-branch mixing blocks (`MMB`)
- Direction-aware enhancement block (`PDEGConv`)
- Cross-scale adaptive fusion (`CASADSF`)
- IoU-aware query ranking in decoder-side selection

Implementation mapping (claim -> code):
- `FSRA`: `src/models/modules.py` (`class FSRA`), used in `src/models/mp_rtdetr.py`
- `MMB`: `src/models/modules.py` (`class MMB`), used in `src/models/mp_rtdetr.py`
- `PDEGConv`: `src/models/modules.py` (`class PDEGConv`), used in `src/models/mp_rtdetr.py`
- `CASADSF`: `src/models/modules.py` (`class CASADSF`), used in `src/models/mp_rtdetr.py`
- IoU-aware query ranking: `src/models/mp_rtdetr.py` (`self.iou_head`, score fusion, `topk` query selection)

## 1. Project Layout

```text
MP-RTDETR/
├── configs/                 # yaml configs
├── scripts/                 # dataset split and label conversion tools
├── src/
│   ├── models/              # model + custom modules + loss
│   ├── datasets/            # dataset reader and augment
│   ├── utils/               # metrics, nms, logger, viz
│   └── postprocess.py
├── tools/                   # train / val / demo / export entrypoints
├── assets/                  # small demo-only assets
├── requirements.txt
└── README.md
```

## 2. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended:
- Python >= 3.10
- PyTorch >= 2.1
- CUDA-capable GPU for training

## 3. Dataset Preparation

This repository does **not** include dataset files or annotations.

Official download pages:
- VisDrone: http://aiskyeye.com/
- DroneVehicle: https://github.com/VisDrone/DroneVehicle

Expected local layout (example):

```text
data/
└── VisDrone/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

Then update paths in `configs/mp_rtdetr_uav.yaml` using **relative paths only**.

## 4. Training

```bash
python tools/train.py --cfg configs/mp_rtdetr_uav.yaml
```

Default outputs:
- checkpoint: `outputs/exp/last.pt`
- log: `outputs/exp/train.log`

## 5. Evaluation

```bash
python tools/val.py --cfg configs/mp_rtdetr_uav.yaml --ckpt outputs/exp/last.pt
```

Current validation script reports a lightweight `mAP@0.50` for fast iteration.

## 6. Inference Demo

```bash
python tools/demo_infer.py \
  --cfg configs/mp_rtdetr_uav.yaml \
  --ckpt outputs/exp/last.pt \
  --source assets/demo.jpg \
  --out outputs/demo
```

## 7. Export (ONNX)

```bash
python tools/export.py \
  --cfg configs/mp_rtdetr_uav.yaml \
  --ckpt outputs/exp/last.pt \
  --out outputs/export/mp_rtdetr.onnx
```

## 8. Data Utilities

Split one image/label folder into train/val:

```bash
python scripts/split_dataset.py \
  --images data/raw/images \
  --labels data/raw/labels \
  --out data/VisDrone \
  --train-ratio 0.9
```

Convert VOC XML labels to YOLO TXT labels:

```bash
python scripts/convert_voc_to_yolo.py \
  --xml-dir data/voc/Annotations \
  --out-dir data/voc/yolo_labels \
  --classes data/voc/classes.txt
```

## 9. Model Weights

Pretrained weights are not distributed in this repository.

An external download link for released checkpoints will be provided after paper acceptance. Until then, please train from scratch with the scripts above.

## 10. Repro Notes

- Fix seeds in config (`seed`) for repeatable runs.
- Start with smaller `input_size` or `batch_size` if GPU memory is limited.
- Keep `num_workers` consistent when comparing speed.

## 11. Citation

```bibtex
@article{mp_rtdetr_2026,
  title={MP-RTDETR: Multi-scale Pinwheel-Enhanced RT-DETR for Small and Dense Object Detection},
  journal={Journal of King Saud University - Computer and Information Sciences},
  year={2026},
  note={Under review}
}
```

## 12. License

Released under Apache-2.0. See `LICENSE` for full text.

## 13. Reviewer Quick Check

Minimal sanity path for reviewers:

```bash
# 1) install
pip install -r requirements.txt

# 2) run lightweight CI-equivalent local checks
python -m py_compile tools/train.py tools/val.py tools/demo_infer.py tools/export.py
python -m py_compile src/models/mp_rtdetr.py src/models/modules.py src/models/losses.py

# 3) quick forward smoke
python - << 'PY'
import torch
from src.models import MPRTDETR
m = MPRTDETR(num_classes=10, width=256, num_queries=100)
y = m(torch.randn(1,3,640,640))
print(y['pred'].shape)
PY
```

## 14. FAQ

**Q: Why are there no datasets or checkpoints in this repo?**  
A: To keep the repository lightweight, compliant, and fully public-safe.

**Q: Can I use absolute paths in configs?**  
A: Avoid absolute paths. Use relative paths so experiments are portable.
