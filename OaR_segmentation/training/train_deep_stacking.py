import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')


from OaR_segmentation.utilities.concat_output_prediction import create_combined_dataset
from OaR_segmentation.training.trainers.ConvolutionTrainer import ConvolutionTrainer
from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.utilities.paths import Paths
from copy import deepcopy


if __name__ == "__main__":
    
    load_dir_list = {
        "1": "10018/model_best.model",
        "2": "10011/model_best.model",
        "3": "10025/model_best.model",
        "4": "10040/model_best.model",
        "5": "10015/model_best.model",
        "6": "10034/model_best.model",
        "coarse": "931/model_best.model"
    }

    models = {
        "1": "seresunet",
        "2": "unet",
        "3": "unet",
        "4": "seresunet",
        "5": "unet",
        "6": "unet",
        "coarse": "stack_unet"
    }

    labels = {
        #"0": "Bg",
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local" #local, gradient, polimi
    db_prediction_creation = False
    n_classes = 7   # 1 if binary, n+1 if n organ
    scale = 1
    paths = Paths(db=db_name, platform=platform)
    loss_criterion = 'crossentropy' # dice, focal, crossentropy, dc_ce, twersky, jaccard
    lr = 1e-3 
    patience = 5
    deep_supervision = False
    dropout = True
    fine_tuning = False
    batch_size = 1
    scale = 1
    augmentation = False
    feature_extraction = False
    epochs = 500
    validation_size = 0.2
    multi_loss_weights=[1, 1] # [ce, dice]
    channels = 6
    finder_lr_iterations = 2000
    find_optimal_lr = False
    finder_lr_iterations = 2000
    optimizer = "adam" #adam, rmsprop
    telegram = False
    mod_type = "stack_UNet" # Onex1StackConv_Unet, stack_UNet, LogReg_thresholding

    if mod_type == "Onex1StackConv_Unet" or mod_type == "LogReg_thresholding":
        assert not deep_supervision, "Onex1StackConv_Unet/LogReg_thresholding doesn't have deep supervision" 


    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])

        nets[label] = build_net(model=models[label], n_classes=1, channels=1,
                                load_inference=True, load_dir=paths.dir_pretrained_model)

    if db_prediction_creation:
        create_combined_dataset(nets=nets, scale=scale, paths=paths, labels=labels)


    net = build_net(model=mod_type, n_classes=n_classes, channels=channels, load_inference=False, deep_supervision=deep_supervision)

    trainer = ConvolutionTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criterion, val_percent=validation_size,
                                labels=labels, network=net, deep_supervision=deep_supervision, 
                                lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, 
                                dataset_name=db_name, optimizer_type=optimizer, stacking=True, telegram=telegram)


    if find_optimal_lr:
        trainer_temp = deepcopy(trainer)
        trainer_temp.initialize()
        _, _, optimal_lr = trainer_temp.find_lr(num_iters=finder_lr_iterations)
        trainer.lr = optimal_lr

    trainer.initialize()
    trainer.setup_info_dict(dropout=dropout, feature_extraction=feature_extraction, pretrained_model=load_dir_list, 
                            fine_tuning=fine_tuning, used_output_models=models)
    trainer.run_training()
