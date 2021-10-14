import torch
from torch import nn
import torch.nn.functional as F


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Onex1StackConv_Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, deep_supervision=False):
        super(Onex1StackConv_Unet, self).__init__()
        self.deep_supervision = deep_supervision
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = "1x1StackConv_Unet"

        self.outc = OutConv(self.n_channels, self.n_classes)

    def forward(self, x):
        x0 = self.outc(x)
        return x0



def build_Onex1StackConv_Unet(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference,
               deep_supervision):

    if load_inference:
        net = Onex1StackConv_Unet(n_channels=channels, n_classes=n_classes, bilinear=True).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = Onex1StackConv_Unet(n_channels=channels, n_classes=n_classes, bilinear=True, deep_supervision=deep_supervision).cuda()

    return net.to(device=device)
