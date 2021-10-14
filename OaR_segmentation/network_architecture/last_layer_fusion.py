""" Full assembly of the parts to form the complete network """
import torch
from torch import nn
import torch.nn.functional as F
from OaR_segmentation.network_architecture.unet import OutConv



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LastLayerFusionNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_labels):
        super(LastLayerFusionNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.name = "LastLayerFusionNet"
        self.deep_supervision = False
        self.n_labels = n_labels
        
        self.nets=None
        self.outc = OutConv(64*(self.n_labels), n_classes) # fusion last layers

    def initialize(self, nets):
        self.nets=nn.ModuleDict(nets)# dict {1:net, 2:net, ..}


    def forward(self, x):        
        c = None
        for organ in x.keys():
            if self.nets[organ].name == "DeepLab V3":
                t2 = self.nets[organ].decoder_output
            else:
                t2 = self.nets[organ](x[organ])
            
            c = t2 if c is None else torch.cat([c, t2], dim=1)
       
        x0 = self.outc(c)  
        return x0
        

     

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def build_LastLayerFusionNet(channels, n_classes, nets, device, retrain_list, n_labels, load_dir=None):
    # INFERENCE 
    if load_dir is not None:
        model = LastLayerFusionNet(n_channels=channels, n_classes=n_classes, n_labels=n_labels).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        model.initialize(nets)
        model.load_state_dict(ckpt['state_dict'])
    
    # TRAINING
    else:
        for net in nets.keys():
            if not retrain_list[net]:
                set_parameter_requires_grad(nets[net])

        model = LastLayerFusionNet(n_channels=channels, n_classes=n_classes, n_labels=n_labels).cuda()
        model.initialize(nets)

    return model.to(device=device)
