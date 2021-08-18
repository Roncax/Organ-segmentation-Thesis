import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# sys.path.append('..')
from OaR_segmentation.network_architecture.deeplab.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from OaR_segmentation.network_architecture.deeplab.aspp import build_aspp
from OaR_segmentation.network_architecture.deeplab.decoder import build_decoder
from OaR_segmentation.network_architecture.deeplab.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone='resnet', output_stride=16,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def build_deeplabV3(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes,
                    load_inference, backbone):
    if finetuning or feature_extraction:
        net = DeepLab(backbone=backbone, output_stride=16, num_classes=old_classes).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if feature_extraction:
            set_parameter_requires_grad(net)

        if net.sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d
        net.decoder.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              BatchNorm(256),
                                              nn.ReLU(),
                                              nn.Dropout(0.5),
                                              nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                              BatchNorm(256),
                                              nn.ReLU(),
                                              nn.Dropout(0.1),
                                              nn.Conv2d(256, n_classes, kernel_size=1, stride=1))

    elif load_inference:
        net = DeepLab(backbone=backbone, output_stride=16, num_classes=n_classes).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = DeepLab(backbone=backbone, output_stride=16, num_classes=n_classes).cuda()

    net.name = f"DeepLab V3 {backbone}"
    net.n_classes = n_classes
    net.n_channels = channels

    return net.to(device=device)
