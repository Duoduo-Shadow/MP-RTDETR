import argparse
import random
from pathlib import Path


IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=str, required=True)
    ap.add_argument('--labels', type=str, required=True)
    ap.add_argument('--out', type=str, default='data/split')
    ap.add_argument('--train-ratio', type=float, default=0.9)
    ap.add_argument('--seed', type=int, default=42)
    return ap.parse_args()


def _copy_pairs(files, src_img, src_lb, dst_img, dst_lb):
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lb.mkdir(parents=True, exist_ok=True)
    for p in files:
        q = src_lb / f'{p.stem}.txt'
        if not q.exists():
            continue
        (dst_img / p.name).write_bytes(p.read_bytes())
        (dst_lb / q.name).write_bytes(q.read_bytes())


def main():
    args = parse_args()
    src_img = Path(args.images)
    src_lb = Path(args.labels)
    out = Path(args.out)

    files = [p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXT]
    files.sort()

    random.seed(args.seed)
    random.shuffle(files)

    n_train = int(len(files) * args.train_ratio)
    train_files = files[:n_train]
    val_files = files[n_train:]

    _copy_pairs(train_files, src_img, src_lb, out / 'train' / 'images', out / 'train' / 'labels')
    _copy_pairs(val_files, src_img, src_lb, out / 'val' / 'images', out / 'val' / 'labels')

    print('done')
    print('train:', len(train_files))
    print('val:', len(val_files))


if __name__ == '__main__':
    main()
