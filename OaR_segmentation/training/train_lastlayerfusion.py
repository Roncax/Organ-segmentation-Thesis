import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from copy import deepcopy
from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.training.trainers.ConvolutionTrainer import ConvolutionTrainer
from OaR_segmentation.utilities.paths import Paths



def run_training():

    labels = {
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord",
        "5_a":"Esophagus"
    }

    load_dir_list = {
        "1": "10018/model_best.model",
        "2": "10011/model_best.model",
        "3": "10025/model_best.model",
        "4": "10040/model_best.model",
        "5": "10041/model_best.model",
        "6": "10034/model_best.model",
        "5_a":"10015/model_best.model"
    }

    models_type_list = {
        "1": "seresunet",
        "2": "unet",
        "3": "unet",
        "4": "seresunet",
        "5": "unet",
        "6": "unet",
        "5_a":"unet",
        "coarse": "unet"
    }

    retrain_list = {
        "1": False,
        "2": False,
        "3": False,
        "4": False,
        "5": False,
        "6": False,
        "5_a": False
    }

    loss_crit = "crossentropy"  # dice, focal, crossentropy, dc_ce, twersky, jaccard
    model = "fusion_net"
    # SegTHOR, StructSeg2019_Task3_Thoracic_OAR
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    epochs = 500
    batch_size = 1
    lr = 1e-3
    val = 0.2
    patience = 5
    augmentation = True
    scale = 1
    channels = 1
    multi_loss_weights = [1, 1]  # [ce, dice]
    platform = "local"  # local, gradient, polimi
    n_classes = 7   # 1 if binary, n+1 if n organ
    paths = Paths(db=db_name, platform=platform)
    find_optimal_lr = False
    finder_lr_iterations = 2000
    optimizer = "adam"  # adam, rmsprop
    telegram = False
    train_with_reduced_db=False

    # Restore all nets
    nets = {}
    for label in labels.keys():
        if 'coarse' in label:
            n_c = n_classes
        else:
            n_c=1
            
        paths.set_pretrained_model(load_dir_list[label])
        nets[label] = build_net(model=models_type_list[label], n_classes=n_c, channels=1,
                                load_dir=paths.dir_pretrained_model, load_inference=True, lastlayer_fusion=True)

    # BEGIN TRAINING
    net = build_net(model=model, n_classes=n_classes,
                    channels=channels, nets=nets, retrain_list=retrain_list, n_labels=len(labels))

    trainer = ConvolutionTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                 batch_size=batch_size, loss_criterion=loss_crit, val_percent=val,
                                 labels=labels, network=net, lr=lr, patience=patience, epochs=epochs,
                                 multi_loss_weights=multi_loss_weights, platform=platform,
                                 dataset_name=db_name, optimizer_type=optimizer, telegram=telegram, lastlayer_fusion=True, train_with_reduced_db=train_with_reduced_db)

    if find_optimal_lr:
        trainer_temp = deepcopy(trainer)
        trainer_temp.initialize()
        _, _, optimal_lr = trainer_temp.find_lr(
            num_iters=finder_lr_iterations, paths=paths)
        trainer.lr = optimal_lr

    trainer.initialize()
    trainer.setup_info_dict(pretrained_model=load_dir_list,
                            retrain_models=retrain_list)
    trainer.run_training()


if __name__ == '__main__':
    run_training()
