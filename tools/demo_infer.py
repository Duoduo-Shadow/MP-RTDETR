import argparse
from pathlib import Path

import cv2
import torch
import yaml

from src.models import MPRTDETR
from src.postprocess import postprocess
from src.utils.viz import draw_dets, save_vis


# quick visual debug for paper demo

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default='configs/mp_rtdetr_uav.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--source', type=str, required=True)
    ap.add_argument('--out', type=str, default='outputs/demo')
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    model = MPRTDETR(
        num_classes=cfg['num_classes'],
        width=cfg['model']['width'],
        num_queries=cfg['model']['num_queries'],
        edge_gate=cfg['model'].get('edge_gate', True),
        iou_aware_query=cfg['model'].get('iou_aware_query', True),
    ).to(device)

    ck = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ck['model'], strict=True)
    model.eval()

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f'bad image: {src}')

    inp = cv2.cvtColor(cv2.resize(img, (cfg['input_size'], cfg['input_size'])), cv2.COLOR_BGR2RGB)
    inp = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    inp = inp.to(device)

    with torch.no_grad():
        out = model(inp)
        det = postprocess(out, conf_thres=0.25, iou_thres=0.5)[0].cpu()

    vis = draw_dets(img, det)
    save_vis(out_dir / f'{src.stem}_pred.jpg', vis)
    print('saved:', out_dir / f'{src.stem}_pred.jpg')


if __name__ == '__main__':
    main()
