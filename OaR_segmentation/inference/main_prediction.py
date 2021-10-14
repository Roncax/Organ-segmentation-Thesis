import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: /notebooks/Organ-segmentation-Thesis/

import json

from OaR_segmentation.inference.predictors.StackingConvPredictor import StackingConvPredictor
from OaR_segmentation.inference.predictors.StackingArgmaxPredictor import StackingArgmaxPredictor
from OaR_segmentation.inference.predictors.StandardPredictor import StandardPredictor
from OaR_segmentation.inference.predictors.LastlayerPredictor import LastLayerPredictor
from OaR_segmentation.utilities.paths import Paths


if __name__ == "__main__":
    
    load_models_dir = {
        "1": "10018/model_best.model",
        "2": "10011/model_best.model",
        "3": "10025/model_best.model",
        "4": "10040/model_best.model",
        "5": "10015/model_best.model",
        "6": "10034/model_best.model",
    }
    
    models_type_list = {
        "1": "seresunet",
        "2": "unet",
        "3": "unet",
        "4": "seresunet",
        "5": "unet",
        "6": "unet",
                        }

    labels = {
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord",
    }
    
    predict_labels = {
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local"  # local, colab, polimi
    load_dir_metamodel = "1405/model_best.model"
    n_classes = 7   # 1 if binary, n+1 if n organ
    scale = 1
    mask_threshold = 0.5
    channels = 1
    metrics_list =['Dice', 'Precision', 'Recall',  'Avg. Surface Distance', 'Hausdorff Distance 95'] 
    stacking_mode = "convolutional" #none, argmax, convolutional, lastlayer_fusion
    convolutional_meta_type = "stack_UNet" # Onex1StackConv_Unet, stack_UNet, LogReg_thresholding
    logistic_regression_weights = False
    logistic_regression_dir = '1/best_model.model'
    
    # PATHS management
    paths = Paths(db=db_name, platform=platform)
    dict_inference_results = json.load(open(paths.json_file_inference_results))
    dict_db_info = json.load(open(paths.json_file_database))
    experiment_num = json.load(open(paths.json_experiments_settings))["inference_experiment"] + 1
    paths.dir_logreg = paths.dir_logreg + f'/{logistic_regression_dir}'

    # used for results storage purpose
    dict_test_info = {
        "db": db_name,
        "fusion_model": load_dir_metamodel,
        "used_models": load_models_dir,
        "scale": scale,
        "mask_threshold": mask_threshold,
        "segmentation_models": models_type_list,
        "stacking_mode": stacking_mode,
        "labels": labels,
        "logistic_regression_weights": logistic_regression_dir if logistic_regression_weights else 'NA',
        "convolutional_meta_type": convolutional_meta_type if stacking_mode == 'convolutional' else 'NA'
    }

    ######## BEGIN INFERENCE ########
    # standard multiclass
    if stacking_mode == "none":
        predictor = StandardPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                      labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, models_type_list=models_type_list)
       
    elif stacking_mode == "argmax":
        predictor = StackingArgmaxPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                            labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, models_type_list=models_type_list)
        
    elif stacking_mode == "convolutional":
        predictor = StackingConvPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                          labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, load_dir_metamodel=load_dir_metamodel, models_type_list=models_type_list, meta_net_model=convolutional_meta_type)
    
    elif stacking_mode == "lastlayer_fusion":
        predictor = LastLayerPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                       labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, load_dir_metamodel=load_dir_metamodel, models_type_list=models_type_list)
        
    predictor.predict()
    predictor.compute_save_metrics(metrics_list=metrics_list, db_name=db_name, colormap=dict_db_info["colormap"], 
                                   experiment_num=experiment_num, dict_test_info=dict_test_info,
                                   labels=predict_labels)
        
