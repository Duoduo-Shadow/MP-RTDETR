import argparse
from pathlib import Path

import torch
import yaml

from src.models import MPRTDETR


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default='configs/mp_rtdetr_uav.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--out', type=str, default='outputs/export/mp_rtdetr.onnx')
    ap.add_argument('--opset', type=int, default=12)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = MPRTDETR(
        num_classes=cfg['num_classes'],
        width=cfg['model']['width'],
        num_queries=cfg['model']['num_queries'],
        edge_gate=cfg['model'].get('edge_gate', True),
        iou_aware_query=cfg['model'].get('iou_aware_query', True),
    )

    ck = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()

    size = int(cfg['input_size'])
    dummy = torch.randn(1, 3, size, size)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=['images'],
        output_names=['pred', 'score'],
        dynamic_axes={
            'images': {0: 'batch'},
            'pred': {0: 'batch'},
            'score': {0: 'batch'},
        },
    )

    print('exported:', out_path)


if __name__ == '__main__':
    main()
