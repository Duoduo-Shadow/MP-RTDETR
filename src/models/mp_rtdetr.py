import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import ConvBNAct, FSRA, MMB, PDEGConv, CASADSF


class TinyBackbone(nn.Module):
    def __init__(self, w=256):
        super().__init__()
        c1, c2, c3 = w // 4, w // 2, w
        self.stem = ConvBNAct(3, c1, 3, 2)
        self.stage2 = nn.Sequential(ConvBNAct(c1, c2, 3, 2), ConvBNAct(c2, c2, 3, 1))
        self.stage3 = nn.Sequential(ConvBNAct(c2, c3, 3, 2), ConvBNAct(c3, c3, 3, 1))
        self.stage4 = nn.Sequential(ConvBNAct(c3, c3, 3, 2), ConvBNAct(c3, c3, 3, 1))

    def forward(self, x):
        x = self.stem(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class DetHead(nn.Module):
    def __init__(self, c, nc):
        super().__init__()
        self.pred = nn.Conv2d(c, nc + 5, 1)

    def forward(self, x):
        return self.pred(x)


class MPRTDETR(nn.Module):
    def __init__(self, num_classes=10, width=256, num_queries=300, edge_gate=True, iou_aware_query=True):
        super().__init__()
        self.nc = num_classes
        self.num_queries = num_queries
        self.iou_aware_query = iou_aware_query

        self.backbone = TinyBackbone(width)
        c3, c4, c5 = width // 2, width, width

        self.fsra3 = FSRA(c3)
        self.fsra4 = FSRA(c4)
        self.fsra5 = FSRA(c5)

        self.mmb3 = MMB(c3)
        self.mmb4 = MMB(c4)
        self.mmb5 = MMB(c5)

        self.pdeg3 = PDEGConv(c3) if edge_gate else nn.Identity()
        self.pdeg4 = PDEGConv(c4) if edge_gate else nn.Identity()
        self.pdeg5 = PDEGConv(c5) if edge_gate else nn.Identity()

        self.fuse4 = CASADSF(c4)
        self.fuse3 = CASADSF(c3)

        self.lat5 = ConvBNAct(c5, c4, 1, 1)
        self.lat4 = ConvBNAct(c4, c3, 1, 1)

        self.head3 = DetHead(c3, num_classes)
        self.head4 = DetHead(c4, num_classes)
        self.head5 = DetHead(c5, num_classes)

        self.iou_head = nn.Sequential(nn.Linear(num_classes + 5, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def _decode_level(self, p):
        b, c, h, w = p.shape
        p = p.permute(0, 2, 3, 1).contiguous()  # B,H,W,C

        box = torch.sigmoid(p[..., :4])
        obj = torch.sigmoid(p[..., 4:5])
        cls = torch.sigmoid(p[..., 5:])

        yy, xx = torch.meshgrid(torch.arange(h, device=p.device), torch.arange(w, device=p.device), indexing='ij')
        grid = torch.stack((xx, yy), dim=-1).float().unsqueeze(0)
        xy = (box[..., :2] + grid) / torch.tensor([w, h], device=p.device)
        wh = box[..., 2:4]
        box = torch.cat([xy, wh], dim=-1)

        pred = torch.cat([box, obj, cls], dim=-1).view(b, -1, self.nc + 5)
        return pred

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)

        p3 = self.fsra3(self.mmb3(self.pdeg3(p3)))
        p4 = self.fsra4(self.mmb4(self.pdeg4(p4)))
        p5 = self.fsra5(self.mmb5(self.pdeg5(p5)))

        p4_up = F.interpolate(self.lat5(p5), size=p4.shape[-2:], mode='nearest')
        p4 = self.fuse4(p4, p4_up)

        p3_up = F.interpolate(self.lat4(p4), size=p3.shape[-2:], mode='nearest')
        p3 = self.fuse3(p3, p3_up)

        out3 = self._decode_level(self.head3(p3))
        out4 = self._decode_level(self.head4(p4))
        out5 = self._decode_level(self.head5(p5))
        pred = torch.cat([out3, out4, out5], dim=1)

        score = pred[..., 4:5] * pred[..., 5:].max(dim=-1, keepdim=True)[0]
        if self.iou_aware_query:
            iou_score = torch.sigmoid(self.iou_head(pred))
            score = 0.5 * score + 0.5 * iou_score

        k = min(self.num_queries, pred.shape[1])
        idx = torch.topk(score.squeeze(-1), k=k, dim=1).indices
        idx = idx.unsqueeze(-1).expand(-1, -1, pred.shape[-1])
        pred = torch.gather(pred, 1, idx)
        score = torch.gather(score, 1, idx[..., :1])

        return {
            "pred": pred,
            "score": score,
        }
