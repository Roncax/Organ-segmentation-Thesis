import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.training.trainers.ConvolutionTrainer import ConvolutionTrainer
from OaR_segmentation.network_architecture.net_factory import build_net
from copy import deepcopy


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
                "1": "10018/model_best.model",
                "2": "1049/model_best.model",
                "3": "1051/model_best.model",
                "4": "1052/model_best.model",
                "5": "1053/model_best.model",
                "6": "1054/model_best.model"
                }
    
    models_type_list = {
                "1": "seresunet",
                "2": "unet",
                "3": "unet",
                "4": "unet",
                "5": "unet",
                "6": "unet",
                "coarse": "unet"
                    }
    
    retrain_list={
            "1": False,
            "2": False,
            "3": False,
            "4": False,
            "5": False,
            "6": False
    }

    loss_crit = "crossentropy" # dice, focal, crossentropy, dc_ce, twersky, jaccard
    model = "fusion_net"  
    db_name = "StructSeg2019_Task3_Thoracic_OAR"   #SegTHOR, StructSeg2019_Task3_Thoracic_OAR
    epochs = 500  
    batch_size = 1  
    lr = 1e-3
    val = 0.2  
    patience = 5  
    augmentation = True  
    scale = 1 
    channels = 1 
    multi_loss_weights = [1, 1] # [ce, dice]
    platform = "local"  # local, gradient, polimi
    n_classes = 7   # 1 if binary, n+1 if n organ
    paths = Paths(db=db_name, platform=platform)
    find_optimal_lr = False
    finder_lr_iterations = 2000
    optimizer = "adam" #adam, rmsprop
    telegram = False


    # Restore all nets
    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])
        nets[label] = build_net(model=models_type_list[label], n_classes=1, channels=1, 
                                load_dir=paths.dir_pretrained_model, load_inference=True, lastlayer_fusion=True)

    # BEGIN TRAINING
    net = build_net(model=model, n_classes=n_classes, channels=channels, nets=nets, retrain_list=retrain_list)

    trainer = ConvolutionTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_crit, val_percent=val,
                                labels=labels, network=net, lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, 
                                dataset_name=db_name,optimizer_type=optimizer, telegram=telegram, lastlayer_fusion=True)

    if find_optimal_lr:
        trainer_temp = deepcopy(trainer)
        trainer_temp.initialize()
        _, _, optimal_lr = trainer_temp.find_lr(num_iters=finder_lr_iterations, paths=paths)
        trainer.lr = optimal_lr

    trainer.initialize()
    trainer.setup_info_dict(pretrained_model=load_dir_list)
    trainer.run_training()



if __name__ == '__main__':
    run_training()
