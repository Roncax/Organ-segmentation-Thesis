import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.training.trainers.ConvolutionTrainer import ConvolutionTrainer
from OaR_segmentation.network_architecture.net_factory import build_net
from copy import deepcopy


def run_training():
    
    labels = {
        "1": "Esophagus",
        "2": "Heart",
        "3": "Trachea",
        "4": "Aorta"
        }
    
    load_dir_list = {
                "1": "1048/model_best.model",
                "2": "1049/model_best.model",
                "3": "1051/model_best.model",
                "4": "1052/model_best.model",
                "5": "1053/model_best.model",
                "6": "1054/model_best.model",
                "coarse": "931/model_best.model"
                }

    # dice, focal, crossentropy, dc_ce
    loss_criteria = {
                "1": "dice",
                "2": "dice",
                "3": "dice",
                "4": "dice",
                "5": "dice",
                "6": "dice",
                "coarse": "crossentropy"
                }

    model = "unet"   #seresunet, unet, segnet, deeplabv3
    db_name = "SegTHOR"   #SegTHOR, StructSeg2019_Task3_Thoracic_OAR
    epochs = 500  
    batch_size = 1  
    lr = 1e-2
    val = 0.20  
    patience = 5  
    fine_tuning = False 
    feature_extraction = False  
    augmentation = True  
    deep_supervision = True  #only unet and seresunet
    dropout = True  #deeplav3 builded in, unet and seresunet only (segnet not supported)
    scale = 1 
    channels = 1 # used for multi-channel 3d method (forse problemi con deeplab)
    multi_loss_weights = [1, 1] # [ce, dice]
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception
    platform = "local"  # local, gradient, polimi
    n_classes = 5   # 1 if binary, n+1 if n organ
    old_classes = -1  # args.old_classes - for transfer learning purpose
    paths = Paths(db=db_name, platform=platform)
    find_optimal_lr = False
    finder_lr_iterations = 2000
    optimizer = "adam" #adam, rmsprop
    telegram = False


    # INITIAL CHECKS
    assert not (feature_extraction and fine_tuning), "Finetuning and feature extraction cannot be both active"
    if feature_extraction or fine_tuning: assert old_classes > 0, "Old classes needed to be specified"

    # Binary or Multiclass
    if n_classes == 1:
        label = next(iter(labels.keys()))
        paths.set_pretrained_model(load_dir_list[label])
        loss_criterion = loss_criteria[label]
        labels = {label: labels[label]}
        pretrained_model = load_dir_list[label]
    else:
        paths.set_pretrained_model(load_dir_list["coarse"])
        loss_criterion = loss_criteria["coarse"]
        pretrained_model = load_dir_list["coarse"]


    # BEGIN TRAINING
    net = build_net(model=model, n_classes=n_classes, finetuning=fine_tuning, load_dir=paths.dir_pretrained_model,
                    channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                    dropout=dropout, deep_supervision=deep_supervision, backbone=deeplabv3_backbone)

    trainer = ConvolutionTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criterion, val_percent=val,
                                labels=labels, network=net, deep_supervision=deep_supervision, 
                                lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, 
                                dataset_name=db_name,optimizer_type=optimizer, telegram=telegram)
    

    if find_optimal_lr:
        trainer_temp = deepcopy(trainer)
        trainer_temp.initialize()
        _, _, optimal_lr = trainer_temp.find_lr(num_iters=finder_lr_iterations, paths=paths)
        trainer.lr = optimal_lr

    trainer.initialize()
    trainer.setup_info_dict(dropout=dropout, feature_extraction=feature_extraction, pretrained_model=pretrained_model, fine_tuning=fine_tuning)
    trainer.run_training()



if __name__ == '__main__':
    run_training()
