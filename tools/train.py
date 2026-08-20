import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import YoloDetDataset, collate_fn
from src.models import MPRTDETR, DetectionCriterion
from src.utils.logger import SimpleLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', type=str, default='configs/mp_rtdetr_uav.yaml')
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(int(cfg.get('seed', 42)))
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    ds = YoloDetDataset(
        cfg['dataset']['train_images'],
        cfg['dataset']['train_labels'],
        img_size=cfg['input_size'],
        hflip=cfg['augment'].get('hflip', 0.5),
        hsv=cfg['augment'].get('hsv', 0.4),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = MPRTDETR(
        num_classes=cfg['num_classes'],
        width=cfg['model']['width'],
        num_queries=cfg['model']['num_queries'],
        edge_gate=cfg['model'].get('edge_gate', True),
        iou_aware_query=cfg['model'].get('iou_aware_query', True),
    ).to(device)

    criterion = DetectionCriterion(cfg['num_classes'])
    optim = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get('amp', True)))

    save_dir = Path(cfg.get('save_dir', 'outputs/exp'))
    logger = SimpleLogger(save_dir)

    for ep in range(cfg['epochs']):
        model.train()
        pbar = tqdm(dl, desc=f'ep {ep+1}/{cfg["epochs"]}', ncols=100)

        avg_loss = 0.0
        for i, (imgs, targets, _, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)

            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get('amp', True))):
                out = model(imgs)
                loss_dict = criterion(out, targets)
                loss = loss_dict['loss']

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            avg_loss += float(loss.item())
            if (i + 1) % 10 == 0:
                pbar.set_postfix(loss=f'{avg_loss / (i + 1):.4f}')

        ep_loss = avg_loss / max(1, len(dl))
        logger.log(f'epoch={ep+1} loss={ep_loss:.5f}')

        ckpt = {
            'epoch': ep + 1,
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'cfg': cfg,
        }
        torch.save(ckpt, save_dir / 'last.pt')

        if (ep + 1) % 30 == 0:
            for g in optim.param_groups:
                g['lr'] *= 0.5


if __name__ == '__main__':
    main()
