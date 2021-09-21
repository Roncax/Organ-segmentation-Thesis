import torch
import segmentation_models_pytorch as smp




def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def build_deeplabV3(channels, n_classes, finetuning, load_dir, device, feature_extraction, old_classes,
                    load_inference, lastlayer_fusion):
    if finetuning or feature_extraction:
        net = smp.DeepLabV3(in_channels=1, classes=old_classes, encoder_weights=None).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if feature_extraction:
            set_parameter_requires_grad(net)

        net.segmentation_head = smp.SegmentationHead(
            in_channels=net.decoder.out_channels,
            out_channels=net.classes,
            activation=net.activation,
            kernel_size=1,
            upsampling=net.upsampling,
        )

    elif load_inference:
        net = smp.DeepLabV3(in_channels=1, classes=n_classes, encoder_weights=None).cuda()
        ckpt = torch.load(load_dir, map_location=device)
        net.load_state_dict(ckpt['state_dict'])

    else:
        net = smp.DeepLabV3(in_channels=1, classes=n_classes, encoder_weights=None).cuda()

    net.name = "DeepLab V3"
    net.n_classes = n_classes
    net.n_channels = channels
    net.deep_supervision = False
    net.lastlayer_fusion = lastlayer_fusion

    return net.to(device=device)
