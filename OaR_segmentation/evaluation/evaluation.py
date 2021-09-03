from tqdm import tqdm

import os
import numpy as np
import torch
import torch.nn.functional as F
from OaR_segmentation.utilities.build_volume import build_np_volume
from OaR_segmentation.evaluation.metrics import ConfusionMatrix
import OaR_segmentation.evaluation.metrics as metrics


def eval_train(net, loader, device):
    #Evaluation of the net (multiclass -> crossentropy, binary -> dice)
    net.eval()  # the net is in evaluation mode
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batches
    tot = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    with tqdm(total=n_val, desc='Real validation round', unit='batch', leave=False) as pbar:
        # iterate over all val batch
        for batch in loader:
            imgs = batch['image_coarse']
            true_masks = batch['mask_gt']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            
            if net.deep_supervision:
                mask_pred = mask_pred[0]
                
                # iterate over all files of single batch


            for true_mask, pred in zip(true_masks, mask_pred):
                if net.n_classes > 1:
                    # multiclass evaluation over single image-mask pair
                    tot += F.cross_entropy(input=mask_pred, target=true_masks.squeeze(1)).item()
                else:
                # Single class evaluation over all validation volume
                    pred = torch.sigmoid(pred)
                    pred = (pred > 0.5).float()  # 0 or 1 by threeshold
                    pred=pred.squeeze(dim=1)

                    pred = pred.detach().cpu().numpy()
                    true_mask = true_mask.detach().cpu().numpy()

                    cm = metrics.ConfusionMatrix(test=pred, reference=true_mask)
                    tp_, fp_, tn_, fn_ = cm.get_matrix()
                    tp += tp_
                    fp += fp_
                    tn += tn_
                    fn += fn_

            pbar.update(n=1)

    net.train()  # the net return to training mode
    return 1 - (2 * tp / (2 * tp + fp + fn)) if net.n_classes == 1 else tot / n_val


def eval_inference(patient, mask_dict, paths):
    # build np volume and confusion matrix
    patient_volume = build_np_volume(dir=os.path.join(paths.dir_mask_prediction, patient))
    gt_volume = build_np_volume(dir=os.path.join(paths.dir_test_GTimg, patient))

    organ_results = {}
    for key in mask_dict:
        # select only a specific class volume
        patient_volume_cp = np.zeros(shape=patient_volume.shape)
        patient_volume_cp[patient_volume == float(key)] = 1
        gt_volume_cp = np.zeros(shape=gt_volume.shape)
        gt_volume_cp[gt_volume == float(key)] = 1

        cm = ConfusionMatrix(test=patient_volume_cp, reference=gt_volume_cp)
        organ_results[mask_dict[key]] = cm

    return organ_results
