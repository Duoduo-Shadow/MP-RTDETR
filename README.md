# MP-RTDETR

MP-RTDETR is a real-time detector designed for UAV aerial scenes with small and densely distributed targets. The project focuses on practical improvements over a transformer-based detection baseline by adding scale-aware allocation, directional edge enhancement, and IoU-aware query selection.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.10+ and PyTorch 2.1+.

## 2. Dataset Preparation

This repository does **not** contain any dataset files.

Please download datasets from official sources:
- VisDrone: http://aiskyeye.com/
- DroneVehicle: https://github.com/VisDrone/DroneVehicle

Then convert to YOLO-format labels and place files under your local `data/` directory. Example layout:

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

Update relative paths in `configs/mp_rtdetr_uav.yaml` accordingly.

## 3. Training

```bash
python tools/train.py --cfg configs/mp_rtdetr_uav.yaml
```

## 4. Evaluation

```bash
python tools/val.py --cfg configs/mp_rtdetr_uav.yaml --ckpt outputs/exp/last.pt
```

The validation script reports mAP@0.50 for quick comparison.

## 5. Inference Demo

```bash
python tools/demo_infer.py \
  --cfg configs/mp_rtdetr_uav.yaml \
  --ckpt outputs/exp/last.pt \
  --source assets/demo.jpg \
  --out outputs/demo
```

## 6. Model Weights

Pretrained weights are not stored in this repository.

A downloadable package link will be provided after paper acceptance. Until then, users can train from scratch using the provided scripts.

## 7. Citation

```bibtex
@article{mp_rtdetr_2026,
  title={MP-RTDETR: Multi-scale Pinwheel-Enhanced RT-DETR for Small and Dense Object Detection},
  journal={Journal of King Saud University - Computer and Information Sciences},
  year={2026},
  note={Under review}
}
```

## 8. License

This project is released under the Apache-2.0 License. See `LICENSE` for details.
