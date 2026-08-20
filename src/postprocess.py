import torch

from src.utils.nms import batched_nms_xyxy


def xywh_to_xyxy(box):
    x, y, w, h = box.unbind(-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def postprocess(outputs, conf_thres=0.25, iou_thres=0.5):
    pred = outputs['pred']
    all_out = []

    for bi in range(pred.shape[0]):
        p = pred[bi]
        obj = p[:, 4:5]
        cls_prob, cls_id = p[:, 5:].max(dim=1)
        score = (obj.squeeze(1) * cls_prob)

        keep = score > conf_thres
        if keep.sum() == 0:
            all_out.append(torch.zeros((0, 6), device=pred.device))
            continue

        box = xywh_to_xyxy(p[keep, :4])
        s = score[keep]
        c = cls_id[keep]

        keep_idx = batched_nms_xyxy(box, s, c, iou_thres=iou_thres)
        out = torch.cat([box[keep_idx], s[keep_idx, None], c[keep_idx, None].float()], dim=1)
        all_out.append(out)

    return all_out
