import torch

from OaR_segmentation.network_architecture.segnet import build_segnet
from OaR_segmentation.network_architecture.unet import build_unet
from OaR_segmentation.network_architecture.stack_testnet import build_stack_unet

from OaR_segmentation.network_architecture.se_resunet import build_SeResUNet
from OaR_segmentation.network_architecture.deeplab_v3p import build_deeplabV3
from OaR_segmentation.network_architecture.last_layer_fusion import build_LastLayerFusionNet
from OaR_segmentation.network_architecture.Onex1ConvNet import build_Onex1StackConv_Unet
from OaR_segmentation.network_architecture.logreg_thresholding import build_LogReg_thresholding

# create a net for every specified model
def build_net(model, channels, n_classes, in_features = None, finetuning=False, load_dir=None, feature_extraction=False,
              old_classes=None, load_inference=False, dropout=False, deep_supervision=False, nets=None, 
              lastlayer_fusion=False, retrain_list = None, n_labels=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model == "unet":
        net = build_unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision, 
                         lastlayer_fusion=lastlayer_fusion)
    elif model == "seresunet":
        net = build_SeResUNet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, dropout=dropout,
                              deep_supervision=deep_supervision, lastlayer_fusion=lastlayer_fusion)
    elif model == "segnet":
        net = build_segnet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                                device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                                load_inference=load_inference, lastlayer_fusion=lastlayer_fusion)

    elif model == "deeplabv3":
        # droupot already present
        net = build_deeplabV3(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                              device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                              load_inference=load_inference, lastlayer_fusion=lastlayer_fusion)

    elif model == "stack_UNet":
        net = build_stack_unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)
        
    elif model == "fusion_net":
        #assert nets is not None, "Fusion with no nets?, try again"
        #assert retrain_list is not None, "Fusion with no retrain_list?, try again"
        net = build_LastLayerFusionNet(channels=channels, n_classes=n_classes, load_dir=load_dir,
                                       device=device, nets=nets, retrain_list=retrain_list, n_labels=n_labels, in_features=in_features)
    elif model == "Onex1StackConv_Unet":
        net = build_Onex1StackConv_Unet(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)
        
    elif model == "LogReg_thresholding":
        net = build_LogReg_thresholding(channels=channels, n_classes=n_classes, finetuning=finetuning, load_dir=load_dir,
                         device=device, feature_extraction=feature_extraction, old_classes=old_classes,
                         load_inference=load_inference, deep_supervision=deep_supervision)

    else:
        net = None
        assert net is not None, "WARNING! The specified net doesn't exist"

    
    return net



