""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
import torch.nn.functional as F



class LastLayerFusionNet(nn.Module):
    def __init__(self, n_channels, n_classes, nets):
        super(LastLayerFusionNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.name = "LastLayerFusionNet"
        self.nets=nets #dict
        


    def forward(self, x):
        # TODO tutta la pipeline per ogni rete (x é un dizionario di differenti immagini preprocessate in modo differente)
        pass
        



def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def build_LastLayerFusionNet(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference,
               deep_supervision):
    
    
    
    if finetuning or feature_extraction:
        net = UNet(n_channels=channels, n_classes=old_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if feature_extraction:
            set_parameter_requires_grad(net)
        net.outc = OutConv(64, n_classes)

    elif load_inference:
        net = UNet(n_channels=channels, n_classes=n_classes, bilinear=True).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = UNet(n_channels=channels, n_classes=n_classes, bilinear=True,
                   deep_supervision=deep_supervision).cuda()

    net.n_classes = n_classes

    return net.to(device=device)
