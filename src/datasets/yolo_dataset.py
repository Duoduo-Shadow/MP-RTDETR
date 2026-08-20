import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class YoloDetDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640, hflip=0.5, hsv=0.4):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.hflip = hflip
        self.hsv = hsv

        exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.files = [p for p in self.img_dir.rglob('*') if p.suffix.lower() in exts]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def _load_label(self, img_path):
        label_path = self.label_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            return np.zeros((0, 5), dtype=np.float32)

        rows = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                sp = line.strip().split()
                if len(sp) != 5:
                    continue
                rows.append([float(x) for x in sp])
        if len(rows) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        return np.array(rows, dtype=np.float32)

    def _augment(self, img, labels):
        if random.random() < self.hflip:
            img = img[:, ::-1].copy()
            if labels.shape[0] > 0:
                labels[:, 1] = 1.0 - labels[:, 1]

        if random.random() < self.hsv:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-8, 8)) % 180
            hsv[..., 1] *= random.uniform(0.8, 1.2)
            hsv[..., 2] *= random.uniform(0.8, 1.2)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return img, labels

    def __getitem__(self, idx):
        p = self.files[idx]
        img = cv2.imread(str(p))
        if img is None:
            raise RuntimeError(f'bad img: {p}')

        labels = self._load_label(p)
        img, labels = self._augment(img, labels)

        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # label format for criterion: [batch_idx, cls, x, y, w, h]
        tgt = torch.zeros((labels.shape[0], 6), dtype=torch.float32)
        if labels.shape[0] > 0:
            tgt[:, 1:] = torch.from_numpy(labels)

        return img, tgt, (h0, w0), p.name


def collate_fn(batch):
    imgs, targets, shapes, names = zip(*batch)
    imgs = torch.stack(imgs, 0)

    cat_t = []
    for bi, t in enumerate(targets):
        if t.numel() == 0:
            continue
        tc = t.clone()
        tc[:, 0] = bi
        cat_t.append(tc)

    if len(cat_t) > 0:
        targets = torch.cat(cat_t, 0)
    else:
        targets = torch.zeros((0, 6), dtype=torch.float32)

    return imgs, targets, shapes, names
