import torch


def box_iou_xyxy(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]).clamp(0) * (box1[:, 3] - box1[:, 1]).clamp(0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(0) * (box2[:, 3] - box2[:, 1]).clamp(0)

    lt = torch.max(box1[:, None, :2], box2[:, :2])
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter + 1e-6
    return inter / union


def nms_single(boxes, scores, iou_thres=0.5):
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou_xyxy(boxes[i:i + 1], boxes[order[1:]]).squeeze(0)
        order = order[1:][iou <= iou_thres]

    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


def batched_nms_xyxy(boxes, scores, labels, iou_thres=0.5):
    keep_all = []
    for c in labels.unique():
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        k = nms_single(boxes[idx], scores[idx], iou_thres)
        keep_all.append(idx[k])

    if len(keep_all) == 0:
        return torch.zeros((0,), device=boxes.device, dtype=torch.long)
    keep_all = torch.cat(keep_all)
    keep_all = keep_all[scores[keep_all].argsort(descending=True)]
    return keep_all
