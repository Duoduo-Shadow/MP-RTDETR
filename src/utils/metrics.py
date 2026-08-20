import torch

from .nms import box_iou_xyxy


def _xywh_to_xyxy(box):
    x, y, w, h = box.unbind(-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def evaluate_map50(pred_list, target_list, num_classes=10):
    # simple AP50 for quick validation, not COCO full protocol
    ap_list = []

    for cls_id in range(num_classes):
        preds = []
        n_gt = 0

        for img_idx, (pred, tgt) in enumerate(zip(pred_list, target_list)):
            if tgt.numel() > 0:
                t = tgt[tgt[:, 1] == cls_id]
                n_gt += t.shape[0]
            else:
                t = tgt

            if pred.numel() > 0:
                p = pred[pred[:, 5] == cls_id]
                for row in p:
                    preds.append((img_idx, float(row[4]), row[:4]))

        if n_gt == 0:
            continue
        if len(preds) == 0:
            ap_list.append(0.0)
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        tp = torch.zeros(len(preds))
        fp = torch.zeros(len(preds))

        matched = {}
        for i, (img_idx, _, pbox) in enumerate(preds):
            tgt = target_list[img_idx]
            tgt = tgt[tgt[:, 1] == cls_id]
            if tgt.numel() == 0:
                fp[i] = 1
                continue

            gt_box = _xywh_to_xyxy(tgt[:, 2:6])
            iou = box_iou_xyxy(pbox.unsqueeze(0), gt_box).squeeze(0)
            best_i = int(iou.argmax())
            best_v = float(iou[best_i])

            key = (img_idx, best_i)
            if best_v >= 0.5 and key not in matched:
                tp[i] = 1
                matched[key] = 1
            else:
                fp[i] = 1

        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recall = tp_cum / (n_gt + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        # 11-point interpolation
        ap = 0.0
        for r in torch.linspace(0, 1, 11):
            p = precision[recall >= r]
            ap += float(p.max()) if p.numel() else 0.0
        ap /= 11.0
        ap_list.append(ap)

    if len(ap_list) == 0:
        return 0.0
    return float(sum(ap_list) / len(ap_list))
