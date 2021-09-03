import torch
from torch import nn
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
import segmentation_models_pytorch.losses as losses

def build_loss(loss_criterion, deep_supervision, n_classes, class_weights=None, ce_dc_weights=None):
    
    if class_weights is not None:
        weights = (torch.FloatTensor(class_weights).to(device="cuda").unsqueeze(dim=0))

    mode = losses.constants.BINARY_MODE if n_classes == 1 else losses.constants.MULTICLASS_MODE

    switcher = {
        "dice": losses.DiceLoss(mode=mode),
        "crossentropy": nn.CrossEntropyLoss(weight=weights) if mode=="multiclass" else nn.BCEWithLogitsLoss(),
        "focal": losses.FocalLoss(mode=mode),
        "dc_ce": BCE_DC_loss(ce_dc_weights=ce_dc_weights, mode = mode, class_weights=weights)
    }

    loss = switcher.get(loss_criterion, "Error, the specified criterion doesn't exist")

    if deep_supervision:
        loss = MultipleOutputLoss2(loss, loss_type=loss_criterion)

    return loss




class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, loss_type, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc). Used for DEEP SUPERVISION
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss
        self.loss_type = loss_type

    def forward(self, x, y):
        
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y)

        for i in range(1, len(x)):
            if weights[i] != 0: 
                l += weights[i] * self.loss(x[i], y)
        return l

class BCE_DC_loss(nn.Module):

    def __init__(self,ce_dc_weights, mode, class_weights):
        super(BCE_DC_loss, self).__init__()

        self.weight_ce = ce_dc_weights[0]
        self.weight_dice = ce_dc_weights[1]

        self.ce = nn.CrossEntropyLoss(weight=class_weights) if mode=="multiclass" else nn.BCEWithLogitsLoss()
        self.dc = losses.DiceLoss(mode=mode)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result
