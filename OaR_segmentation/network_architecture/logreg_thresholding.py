import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
import torch


class LogReg_thresholding(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, deep_supervision=False):
        super(LogReg_thresholding, self).__init__()
        self.deep_supervision = deep_supervision
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.name = "LogReg_thresholding"
        
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor([1.])) for i in range(self.n_classes)])
        self.biases = nn.ParameterList([nn.Parameter(torch.tensor([1.])) for i in range(self.n_classes)])   

    def forward(self, x):
        y = torch.zeros(1, 512, 512, device='cuda:0')
        x = torch.squeeze(x)
        for i, input_image in enumerate(x):
            y = torch.cat((y, torch.unsqueeze(input_image * self.weights[i] + self.biases[i], dim=0)))

            
        return torch.unsqueeze(y, dim=0)



def build_LogReg_thresholding(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes, load_inference,
               deep_supervision):

    if load_inference:
        net = LogReg_thresholding(n_channels=channels, n_classes=n_classes, bilinear=True).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        for n in range(n_classes):
            print(net.weights[n])
            print(net.biases[n])

    else:
        net = LogReg_thresholding(n_channels=channels, n_classes=n_classes, bilinear=True, deep_supervision=deep_supervision).cuda()

    return net.to(device=device)
