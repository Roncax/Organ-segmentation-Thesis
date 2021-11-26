import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: /notebooks/Organ-segmentation-Thesis/

import json

from OaR_segmentation.inference.predictors.StackingConvPredictor import StackingConvPredictor
from OaR_segmentation.inference.predictors.StackingArgmaxPredictor import StackingArgmaxPredictor
from OaR_segmentation.inference.predictors.StandardPredictor import StandardPredictor
from OaR_segmentation.inference.predictors.LastlayerPredictor import LastLayerPredictor
from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.utilities.populate_csv import populate_csv


if __name__ == "__main__":
    
    load_models_dir = {
        "1": "10070/model_best.model",
        "2": "10072/model_best.model",
        "3": "10051/model_best.model",
        "4": "10052/model_best.model",
        "5": "10071/model_best.model",
        "6": "10073/model_best.model",
        "coarse": "10106/model_best.model",
        "4_a": "10090/model_best.model",
        "5_a": "10053/model_best.model"
        }
    
    models_type_list = {
        "1": "seresunet",
        "2": "seresunet",
        "3": "deeplabv3",
        "4": "deeplabv3",
        "5": "seresunet",
        "6": "seresunet",
        "coarse": "deeplabv3",
        "4_a": "unet",
        "5_a": "deeplabv3",
        }

# to effectively the above networks
    labels = {
       "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord",
        #"coarse":"coarse",
        #"4_a": "Trachea",
        #"5_a": "Esophagus"
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
    load_dir_metamodel = "1549/model_best.model"
    n_classes = 7   # 1 if binary, n+1 if n organ
    scale = 1
    mask_threshold = 0.5
    metrics_list =['Dice', 'Precision', 'Recall' ,  'Avg. Surface Distance', 'Hausdorff Distance 95'] 
    stacking_mode = "convolutional" #none, argmax, convolutional, lastlayer_fusion
    convolutional_meta_type = "Onex1StackConv_Unet" # Onex1StackConv_Unet, stack_UNet, LogReg_thresholding
    logistic_regression_weights = False
    logistic_regression_dir = '1/best_model.model'
    in_features = 64*5 + 256*3  # 64 per ogni unet/resnet, 256 per ogni deeplab
    channels = 6 + 7*0 # 1 per ogni rete binaria e 7 per multibinaria
    crop_size = (320,320)

    
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
        "channels":channels,
        "in_features":in_features,
        "mask_threshold": mask_threshold,
        "segmentation_models": models_type_list,
        "stacking_mode": stacking_mode,
        "labels": labels,
        "logistic_regression_weights": logistic_regression_dir if logistic_regression_weights else 'NA',
        "convolutional_meta_type": convolutional_meta_type if stacking_mode == 'convolutional' else 'NA',
        "crop_size":crop_size
    }

    ######## BEGIN INFERENCE ########
    # standard multiclass
    if stacking_mode == "none":
        predictor = StandardPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                      labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights, crop_size=crop_size)
        predictor.initialize(load_models_dir=load_models_dir, channels=1, models_type_list=models_type_list)
       
    elif stacking_mode == "argmax":
        predictor = StackingArgmaxPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                            labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights, crop_size=crop_size)
        predictor.initialize(load_models_dir=load_models_dir, channels=1, models_type_list=models_type_list)
        
    elif stacking_mode == "convolutional":
        predictor = StackingConvPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                          labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights, crop_size=crop_size)
        predictor.initialize(load_models_dir=load_models_dir, channels=channels, load_dir_metamodel=load_dir_metamodel, models_type_list=models_type_list, meta_net_model=convolutional_meta_type)
    
    elif stacking_mode == "lastlayer_fusion":
        predictor = LastLayerPredictor(scale = scale, mask_threshold = mask_threshold,  paths=paths, 
                                       labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights, in_features=in_features, crop_size=crop_size)
        predictor.initialize(load_models_dir=load_models_dir, channels=1, load_dir_metamodel=load_dir_metamodel, models_type_list=models_type_list)
        
    predictor.predict()
    predictor.compute_save_metrics(metrics_list=metrics_list, db_name=db_name, colormap=dict_db_info["colormap"], 
                                   experiment_num=experiment_num, dict_test_info=dict_test_info,
                                   labels=predict_labels, gif_viz=True)
    
    populate_csv(str(experiment_num), predict_labels, metrics_list)
        
