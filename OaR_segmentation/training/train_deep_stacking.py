import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')


from OaR_segmentation.utilities.concat_output_prediction import create_combined_dataset
from OaR_segmentation.training.trainers.ConvolutionTrainer import ConvolutionTrainer
from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.utilities.paths import Paths
from albumentations import augmentations
from copy import deepcopy


if __name__ == "__main__":
    
    load_dir_list = {
        "1": "1048/model_best.model",
        "2": "1049/model_best.model",
        "3": "1051/model_best.model",
        "4": "1052/model_best.model",
        "5": "1053/model_best.model",
        "6": "1054/model_best.model",
        "coarse": "931/model_best.model"
    }

    models = {
        "1": "unet",
        "2": "unet",
        "3": "unet",
        "4": "unet",
        "5": "unet",
        "6": "unet",
        "coarse": "stack_unet"
    }

    labels = {
        "0": "Bg",
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local" #local, gradient, polimi
    db_prediction_creation = True
    n_classes = 7   # 1 if binary, n+1 if n organ
    scale = 1
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception
    channels = 1
    paths = Paths(db=db_name, platform=platform)
    loss_criterion = 'crossentropy' # dice, bce, binaryFocal, multiclassFocal, crossentropy, dc_bce
    lr = 0.02
    patience = 5
    deep_supervision = False
    dropout = False
    fine_tuning = False
    batch_size = 1
    scale = 1
    augmentation = False
    feature_extraction = False
    epochs = 500
    validation_size = 0.2
    multi_loss_weights=[1, 1]
    channels = 6
    find_lr = False
    finder_lr_iterations = 2000
    find_optimal_lr = False
    finder_lr_iterations = 2000
    optimizer = "adam" #adam, rmsprop



    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])
        paths.set_train_stacking_results()

        nets[label] = build_net(model=models[label], n_classes=1, channels=channels,
                                load_inference=True, load_dir=paths.dir_pretrained_model)

    if db_prediction_creation:
        create_combined_dataset(nets=nets, scale=scale, paths=paths, labels=labels)


    net = build_net(model='stack_UNet', n_classes=n_classes, channels=channels, load_inference=False)

    trainer = ConvolutionTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criterion, val_percent=validation_size,
                                labels=labels, network=net, deep_supervision=deep_supervision, 
                                lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, 
                                dataset_name=db_name, optimizer_type=optimizer, stacking=True)

    if find_optimal_lr:
        trainer_temp = deepcopy(trainer)
        trainer_temp.initialize()
        _, _, optimal_lr = trainer_temp.find_lr(num_iters=finder_lr_iterations)
        trainer.lr = optimal_lr

    trainer.initialize()
    trainer.setup_info_dict(dropout=dropout, feature_extraction=feature_extraction, pretrained_model=load_dir_list, fine_tuning=fine_tuning, used_output_models=models)
    trainer.run_training()
