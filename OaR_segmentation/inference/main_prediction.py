import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: /notebooks/Organ-segmentation-Thesis/

import json

from OaR_segmentation.inference.predictors.StackingConvPredictor import StackingConvPredictor
from OaR_segmentation.inference.predictors.StackingArgmaxPredictor import StackingArgmaxPredictor
from OaR_segmentation.inference.predictors.StandardPredictor import StandardPredictor
from OaR_segmentation.utilities.paths import Paths


if __name__ == "__main__":
    
    load_models_dir = {
        "1": "10009/model_best.model",
        "2": "10011/model_best.model",
        "3": "10012/model_best.model",
        "4": "10013/model_best.model",
        "5": "10015/model_best.model",
        "6": "10016/model_best.model",
        "coarse": "1222/model_best.model"
    }
    
    models_type_list = {"1": "unet",
                        "2": "unet",
                        "3": "unet",
                        "4": "unet",
                        "5": "unet",
                        "6": "unet",
                        "coarse": "unet"
                        }

    labels = {
        #"0": "Bg",
        "1": "Esophagus",
        "2": "Heart",
        "3": "Trachea",
        "4": "Aorta"
    }
    
    db_name = "SegTHOR"
    platform = "local"  # local, colab, polimi
    load_dir_metamodel = "189/model_latest.model"
    deeplab_backbone = "mobilenet"  # resnet, drn, mobilenet, xception
    n_classes = 5   # 1 if binary, n+1 if n organ
    scale = 1
    mask_threshold = 0.5
    channels = 1
    metrics_list = ['Precision', 'Recall','Dice', 'Hausdorff Distance 95', 'Avg. Surface Distance']
    stacking_mode = "none" #none, argmax, convolutional
    
    # PATHS management
    paths = Paths(db=db_name, platform=platform)
    dict_inference_results = json.load(open(paths.json_file_inference_results))
    dict_db_info = json.load(open(paths.json_file_database))
    experiment_num = json.load(open(paths.json_experiments_settings))["inference_experiment"] + 1

    # used for results storage purpose
    dict_test_info = {
        "db": db_name,
        "used_models": load_models_dir,
        "scale": scale,
        "mask_threshold": mask_threshold,
        "segmentation_models": models_type_list,
        "stacking_mode": stacking_mode,
        "labels": len(labels)
    }

    ######## BEGIN INFERENCE ########
    # standard multiclass
    if stacking_mode == "none":
        predictor = StandardPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, labels=labels, n_classes=n_classes)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, models_type_list=models_type_list, deeplab_backbone=deeplab_backbone)
       
    elif stacking_mode == "argmax":
        predictor = StackingArgmaxPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, labels=labels, n_classes=n_classes)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, models_type_list=models_type_list)
        
    elif stacking_mode == "convolutional":
        predictor = StackingConvPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, labels=labels, n_classes=n_classes)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, load_dir_metamodel=load_dir_metamodel, models_type_list=models_type_list)
        
        
    predictor.predict()
    predictor.compute_save_metrics(metrics_list=metrics_list, db_name=db_name, colormap=dict_db_info["colormap"], experiment_num=experiment_num, dict_test_info=dict_test_info)
        
