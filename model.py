# fire_damage_pred/model.py
import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3):
        super().__init__()
        pad = k // 2
        self.conv_x = nn.Conv2d(in_ch, 4*hid_ch, k, padding=pad)
        self.conv_h = nn.Conv2d(hid_ch, 4*hid_ch, k, padding=pad)
        self.hid_ch = hid_ch
    def forward(self, x, h, c):
        gates = self.conv_x(x) + self.conv_h(h)
        i,f,o,g = torch.split(gates, self.hid_ch, dim=1)
        i,f,o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f*c + i*g
        h = o*torch.tanh(c)
        return h, c

class FireDamageModel(nn.Module):
    def __init__(self, in_ch, hid_ch, num_cls):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hid_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(hid_ch, num_cls)
    def forward(self, x):        # x: (B, T, C, H, W)
        B,T,C,H,W = x.size()
        h = x.new_zeros(B, self.cell.hid_ch, H, W)
        c = x.new_zeros(B, self.cell.hid_ch, H, W)
        for t in range(T):
            h,c = self.cell(x[:,t], h, c)
        out = self.pool(h).flatten(1)
        return self.fc(out)
