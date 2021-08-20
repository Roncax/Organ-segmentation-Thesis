import torch
from torch import nn
from pytorch_toolbelt.losses.focal import FocalLoss, BinaryFocalLoss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2


def build_loss(loss_criterion, deep_supervision, class_weights=None, bce_dc_weights=None):
    

    weights = (torch.FloatTensor(class_weights).to(device="cuda").unsqueeze(dim=0))

    switcher = {
        "dice": DiceLoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "crossentropy": nn.CrossEntropyLoss(weight=weights),
        "binaryFocal": BinaryFocalLoss(),
        "multiclassFocal": FocalLoss(),
        "dc_bce": BCE_DC_loss(weight_ce=bce_dc_weights[0], weight_dice=bce_dc_weights[1])
    }

    loss = switcher.get(
        loss_criterion, "Error, the specified criterion doesn't exist")

    if deep_supervision:
        loss = MultipleOutputLoss2(loss, loss_type=loss_criterion)

    return loss


class BCE_DC_loss(nn.Module):

    def __init__(self, weight_ce=1, weight_dice=1):
        super(BCE_DC_loss, self).__init__()

        self.weight_ce = weight_ce
        self.weight_dice = weight_dice

        self.ce = nn.BCEWithLogitsLoss()
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result
    
class DiceLoss(nn.Module):
    __name__ = 'dice_loss'
 
    def __init__(self, activation='sigmoid'):
        super(DiceLoss, self).__init__()
        self.activation = activation
 
    def forward(self, y_pr, y_gt):
        return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)
    


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, loss_type, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
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



def diceCoeff(pred, gt, smooth=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
                 raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N
 
 
 
def diceCoeffv2(pred, gt, eps=1e-5, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
                 raise NotImplementedError("Activation implemented for sigmoid and softmax2d activation function operation")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    loss = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return loss.sum() / N