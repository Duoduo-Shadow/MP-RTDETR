import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


# simple converter for common VOC xml format

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml-dir', type=str, required=True)
    ap.add_argument('--out-dir', type=str, required=True)
    ap.add_argument('--classes', type=str, required=True, help='txt file, one class name per line')
    return ap.parse_args()


def voc_box_to_yolo(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    cx = (xmin + xmax) / 2.0 / w
    cy = (ymin + ymax) / 2.0 / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h
    return cx, cy, bw, bh


def main():
    args = parse_args()
    xml_dir = Path(args.xml_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = [x.strip() for x in Path(args.classes).read_text(encoding='utf-8').splitlines() if x.strip()]
    cls2id = {c: i for i, c in enumerate(classes)}

    for xml_path in sorted(xml_dir.glob('*.xml')):
        root = ET.parse(xml_path).getroot()
        size = root.find('size')
        w = float(size.find('width').text)
        h = float(size.find('height').text)

        lines = []
        for obj in root.findall('object'):
            cls = obj.find('name').text.strip()
            if cls not in cls2id:
                continue
            bb = obj.find('bndbox')
            xmin = float(bb.find('xmin').text)
            ymin = float(bb.find('ymin').text)
            xmax = float(bb.find('xmax').text)
            ymax = float(bb.find('ymax').text)
            cx, cy, bw, bh = voc_box_to_yolo((w, h), (xmin, ymin, xmax, ymax))
            lines.append(f'{cls2id[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

        out_path = out_dir / (xml_path.stem + '.txt')
        out_path.write_text('\n'.join(lines), encoding='utf-8')

    print('done:', out_dir)


if __name__ == '__main__':
    main()
