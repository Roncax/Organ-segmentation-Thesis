import numpy as np
import torch
from torch import nn

# return the predicted mask by higher prob value

def combine_predictions(comb_dict, mask_threshold, shape):
    tot = np.zeros(shape)
    for organ1 in comb_dict:
        t = comb_dict[organ1].copy()
        cd = comb_dict.copy()
        cd.pop(organ1)
        for organ2 in cd:
            t[comb_dict[organ1] < comb_dict[organ2]] = 0
        t[t < mask_threshold] = 0
        t[t > mask_threshold] = 1
        tot[t == 1] = organ1
    return tot

def combine_predictions_with_coarse(comb_dict, mask_threshold, shape, coarse):
    tot = np.zeros(shape)
    for organ1 in comb_dict:
        t = comb_dict[organ1].copy()
        cd = comb_dict.copy()
        cd.pop(organ1)
        for organ2 in cd:
            t[comb_dict[organ1] < comb_dict[organ2]] = 0

        t[t < mask_threshold] = 0
        t[t > mask_threshold] = 1
        tot[t == 1] = organ1

    return tot

def test_combine():
    labels = {"0": "Bg",
            "1": "RightLung",
            "2": "LeftLung",
            "3": "Heart",
            "4": "Trachea",
            "5": "Esophagus",
            "6": "SpinalCord"
            }
    mask_threshold = 0.5
    comb_dict = {}
    shape = (20,20)

    for l in labels:

        t = np.round(np.random.rand(20,20), 2)
        comb_dict[l]=t
        print(f"label  {l}")
        print(t)

    tot = combine_predictions_with_coarse(comb_dict=comb_dict, mask_threshold=mask_threshold, shape=shape)
    print(f"Total: \n{tot}")

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

    class DiceLoss(nn.Module):
        __name__ = 'dice_loss'
     
        def __init__(self, activation='sigmoid'):
            super(DiceLoss, self).__init__()
            self.activation = activation
     
        def forward(self, y_pr, y_gt):
            return 1 - diceCoeffv2(y_pr, y_gt, activation=self.activation)
        
        
def test_dice():
    # shape = torch.Size([1, 2, 4, 4])
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 0= background 
    '''
    pred = torch.Tensor([[
        [[0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]],
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]]])
    
    gt = torch.Tensor([[
        [[0, 1, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 1, 1, 0]],
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]],
        [[0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]]]])
    
    
    dice_tumor1 = diceCoeffv2(pred[:, 1:2, :], gt[:, 1:2, :], eps=1, activation=None)
    dice_tumor2 = diceCoeffv2(pred[:, 1:2, :], gt[:, 1:2, :], eps=1e-5, activation=None)
    print('smooth=1 : dice={:.4}'.format(dice_tumor1.item()))
    print('smooth=1e-5 : dice={:.4}'.format(dice_tumor2.item()))
    
    
    # Output result
    # smooth=1 : dice=2.0
    # smooth=1e-5 : dice=2.0
    
test_dice()