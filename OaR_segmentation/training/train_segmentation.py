import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: Organ-segmentation-Thesis

from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.training.trainers.CustomTrainer import CustomTrainer
from OaR_segmentation.network_architecture.net_factory import build_net



def run_training():
    
    labels = {
                "1": "RightLung",
                "2": "LeftLung",
                "3": "Heart",
                "4": "Trachea",
                "5": "Esophagus",
                "6": "SpinalCord"
            }
    
    load_dir_list = {
                    "1": "931/model_best.model",
                    "2": "931/model_best.model",
                    "3": "931/model_best.model",
                    "4": "931/model_best.model",
                    "5": "931/model_best.model",
                    "6": "931/model_best.model",
                    "coarse": ""
                }

    # dice, bce, binaryFocal, multiclassFocal, crossentropy, dc_bce
    loss_criteria = {
                    "1": "dc_bce",
                    "2": "dc_bce",
                    "3": "dc_bce",
                    "4": "dc_bce",
                    "5": "dc_bce",
                    "6": "dc_bce",
                    "coarse": "crossentropy"
                }

    model = "unet"   #seresunet, unet, segnet, deeplabv3
    db_name = "StructSeg2019_Task3_Thoracic_OAR"   #SegTHOR, StructSeg2019_Task3_Thoracic_OAR
    epochs = 500  
    batch_size = 1  
    lr = 0.02  
    val = 0.20  
    patience = 5  
    fine_tuning = False 
    feature_extraction = False  
    augmentation = True  
    train_type = 'coarse'  
    deep_supervision = False  #only unet and seresunet
    dropout = True  #deeplav3 builded in, unet and seresunet only (segnet not supported)
    scale = 1 
    channels = 1 # used for multi-channel 3d method (forse problemi con deeplab)
    multi_loss_weights = [1, 1]  # for composite losses
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception
    platform = "local"  # local, gradient, polimi
    n_classes = 7   # 1 if binary, n+1 if n organ
    old_classes = -1  # args.old_classes - for transfer learning purpose
    paths = Paths(db=db_name, platform=platform)
    find_lr = False

    # INITIAL CHECKS
    assert not (feature_extraction and fine_tuning), "Finetuning and feature extraction cannot be both active"
    if feature_extraction or fine_tuning: assert old_classes > 0, "Old classes needed to be specified"

    # BEGIN TRAINING
    ## COARSE (multiclass) NET
    if train_type == "coarse":
        paths.set_pretrained_model(load_dir_list["coarse"])

        net = build_net(model=model, n_classes=n_classes, finetuning=fine_tuning, load_dir=paths.dir_pretrained_model,
                        channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                        dropout=dropout, deep_supervision=deep_supervision, backbone=deeplabv3_backbone)

        trainer = CustomTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criteria['coarse'], val_percent=val,
                                labels=labels, network=net, deep_supervision=deep_supervision, dropout=dropout,
                                fine_tuning=fine_tuning, feature_extraction=feature_extraction,
                                pretrained_model=load_dir_list["coarse"], lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, dataset_name=db_name)
        trainer.initialize()
        if find_lr:
            trainer.find_lr(num_iters=2000)
        else:
            trainer.run_training()

    ## FINE (multibinary) NETs
    elif train_type == "fine":
        
        # for every label (organ), train a net - can be only 1 organ if
        for label in labels.keys():
            paths.set_pretrained_model(load_dir_list[label])

            net = build_net(model=model, n_classes=1, finetuning=fine_tuning,
                            load_dir=paths.dir_pretrained_model,
                            channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                            dropout=dropout, deep_supervision=deep_supervision, backbone=deeplabv3_backbone)

            trainer = CustomTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                    batch_size=batch_size, loss_criterion=loss_criteria[label], val_percent=val,
                                    labels={label: labels[label]}, network=net, deep_supervision=deep_supervision, dropout=dropout,
                                    fine_tuning=fine_tuning, feature_extraction=feature_extraction,
                                    pretrained_model=load_dir_list[label], lr=lr, patience=patience, epochs=epochs,
                                    multi_loss_weights=multi_loss_weights, platform=platform, dataset_name=db_name)

            trainer.initialize()
            if find_lr:
                trainer.find_lr(num_iters=2000)
            else:
                trainer.run_training()


if __name__ == '__main__':
    run_training()
