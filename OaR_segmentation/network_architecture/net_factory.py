import torch

from OaR_segmentation.network_architecture.segnet import build_segnet
from OaR_segmentation.network_architecture.unet import build_unet
from OaR_segmentation.network_architecture.stack_testnet import build_stack_unet

from OaR_segmentation.network_architecture.se_resunet import build_SeResUNet
from OaR_segmentation.network_architecture.deeplab_v3p import build_deeplabV3


# create a net for every specified model
def build_net(model, channels, n_classes, finetuning=False, load_dir=None, feature_extraction=False,
              old_classes=None, load_inference=False, dropout=False, deep_supervision=False, backbone="resnet"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "unet":
        net = build_unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)
    elif model == "seresunet":
        net = build_SeResUNet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, dropout=dropout,
                              deep_supervision=deep_supervision)
    elif model == "segnet":
        net = build_segnet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                                device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                                load_inference=load_inference)

    elif model == "deeplabv3":
        # droupot already present
        net = build_deeplabV3(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, backbone=backbone)

    elif model == "stack_UNet":
        net = build_stack_unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)

    else:
        net = None
        print("WARNING! The specified net doesn't exist")

    return net



