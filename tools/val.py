import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import YoloDetDataset, collate_fn
from src.models import MPRTDETR
from src.postprocess import postprocess
from src.utils.metrics import evaluate_map50


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default='configs/mp_rtdetr_uav.yaml')
    ap.add_argument('--ckpt', type=str, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    ds = YoloDetDataset(
        cfg['dataset']['val_images'],
        cfg['dataset']['val_labels'],
        img_size=cfg['input_size'],
        hflip=0.0,
        hsv=0.0,
    )
    dl = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'], collate_fn=collate_fn)

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

    pred_all = []
    tgt_all = []

    with torch.no_grad():
        for imgs, targets, _, _ in tqdm(dl, desc='val', ncols=100):
            imgs = imgs.to(device)
            out = model(imgs)
            dets = postprocess(out, conf_thres=0.25, iou_thres=0.5)
            pred_all.extend([d.cpu() for d in dets])

            bsz = imgs.size(0)
            for bi in range(bsz):
                tgt_all.append(targets[targets[:, 0] == bi].cpu())

    map50 = evaluate_map50(pred_all, tgt_all, num_classes=cfg['num_classes'])
    print(f'mAP@0.50: {map50:.4f}')


if __name__ == '__main__':
    main()
