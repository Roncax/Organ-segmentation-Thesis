import torch
from torch import nn
from pytorch_toolbelt.losses.dice import DiceLoss
from pytorch_toolbelt.losses.focal import FocalLoss, BinaryFocalLoss
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from torchgeometry.losses.dice import DiceLoss as DL


def build_loss(loss_criterion, deep_supervision, class_weights=None, bce_dc_weights=None):
    

    weights = (torch.FloatTensor(class_weights).to(device="cuda").unsqueeze(dim=0))

    switcher = {
        "dice": DiceLoss(mode="binary", smooth=1e-5),
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
        self.dc = DiceLoss(mode="binary", smooth=1e-5)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)

        result = self.weight_ce * ce_loss + self.weight_dice * dc_loss

        return result


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
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        # y_temp = []
        # for _ in x:
        #     y_temp.append(y)
        # y = y_temp
        # assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y)

        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y)
        return l
