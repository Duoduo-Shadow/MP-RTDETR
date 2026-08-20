import torch
import torch.nn as nn
import torch.nn.functional as F


def _box_iou_xywh(box1, box2):
    b1_xy = box1[..., :2]
    b1_wh = box1[..., 2:4]
    b2_xy = box2[..., :2]
    b2_wh = box2[..., 2:4]

    b1_xy1 = b1_xy - b1_wh / 2
    b1_xy2 = b1_xy + b1_wh / 2
    b2_xy1 = b2_xy - b2_wh / 2
    b2_xy2 = b2_xy + b2_wh / 2

    inter_xy1 = torch.max(b1_xy1, b2_xy1)
    inter_xy2 = torch.min(b1_xy2, b2_xy2)
    inter_wh = (inter_xy2 - inter_xy1).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]

    a1 = b1_wh[..., 0] * b1_wh[..., 1]
    a2 = b2_wh[..., 0] * b2_wh[..., 1]
    union = a1 + a2 - inter + 1e-6
    return inter / union


class DetectionCriterion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.nc = num_classes
        self.bce = nn.BCELoss(reduction='mean')

    def forward(self, outputs, targets):
        pred = outputs['pred']  # B, Q, 5+nc
        b, q, _ = pred.shape
        obj_pred = pred[..., 4]
        cls_pred = pred[..., 5:]
        box_pred = pred[..., :4]

        obj_t = torch.zeros_like(obj_pred)
        cls_t = torch.zeros_like(cls_pred)
        box_l = torch.tensor(0.0, device=pred.device)
        cls_l = torch.tensor(0.0, device=pred.device)

        for bi in range(b):
            t = targets[targets[:, 0] == bi]
            if t.numel() == 0:
                continue
            gt_cls = t[:, 1].long()
            gt_box = t[:, 2:6]

            iou = _box_iou_xywh(box_pred[bi].unsqueeze(1), gt_box.unsqueeze(0))  # Q,M
            best_q = iou.argmax(dim=0)

            obj_t[bi, best_q] = 1.0
            cls_t[bi, best_q, gt_cls] = 1.0

            box_l = box_l + F.l1_loss(box_pred[bi, best_q], gt_box, reduction='mean')
            cls_l = cls_l + self.bce(cls_pred[bi, best_q], cls_t[bi, best_q])

        obj_l = self.bce(obj_pred, obj_t)
        total = box_l + cls_l + obj_l

        return {
            'loss': total,
            'loss_box': box_l.detach(),
            'loss_cls': cls_l.detach(),
            'loss_obj': obj_l.detach(),
        }
