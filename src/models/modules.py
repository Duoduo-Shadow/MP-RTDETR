import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FSRA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 4)
        self.fc2 = nn.Linear(channels // 4, channels)
        self.spatial = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        ch = F.adaptive_avg_pool2d(x, 1).view(b, c)
        ch = torch.sigmoid(self.fc2(F.relu(self.fc1(ch), inplace=True))).view(b, c, 1, 1)
        sp = torch.sigmoid(self.spatial(x))
        return x * ch * sp


class PDEGConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw = ConvBNAct(c, c, k=3, s=1, g=c)
        self.pw = ConvBNAct(c, c, k=1, s=1)

        sx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        sy = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("sobel_x", sx.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sy.view(1, 1, 3, 3))

    def forward(self, x):
        y = self.pw(self.dw(x))
        g = x.mean(1, keepdim=True)
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
        edge = torch.sigmoid(edge)
        return y * (1.0 + edge)


class CASADSF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.deep_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.shallow_gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c, 1), nn.Sigmoid())
        self.mix = ConvBNAct(c * 2, c, k=1, s=1)

    def forward(self, fd, fs):
        # small trick: keep shallow texture for tiny objs
        fd = fd * self.deep_gate(fd)
        fs = fs * self.shallow_gate(fs)
        out = self.mix(torch.cat([fd, fs], dim=1))
        # print('cas', out.shape)  # debug
        return out
