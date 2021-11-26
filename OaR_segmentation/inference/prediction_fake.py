import json
import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: /notebooks/Organ-segmentation-Thesis/


from torch.utils.data.dataloader import DataLoader
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset

from OaR_segmentation.inference.predictors.StackingConvPredictor import StackingConvPredictor
from OaR_segmentation.inference.predictors.StackingArgmaxPredictor import StackingArgmaxPredictor
from OaR_segmentation.inference.predictors.StandardPredictor import StandardPredictor
from OaR_segmentation.inference.predictors.LastlayerPredictor import LastLayerPredictor
from OaR_segmentation.utilities.paths import Paths
import matplotlib.pyplot as plt

def prediction_plot(exp):
    
    load_models_dir = exp["load_models_dir"]
    models_type_list = exp['models_type_list']

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
    
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local"  # local, colab, polimi
    load_dir_metamodel = exp['load_dir_metamodel']
    n_classes = 7   # 1 if binary, n+1 if n organ
    scale = 1
    mask_threshold = 0.5
    stacking_mode = exp['stacking_mode'] #none, argmax, convolutional, lastlayer_fusion
    convolutional_meta_type = exp['convolutional_meta_type'] # Onex1StackConv_Unet, stack_UNet, LogReg_thresholding
    logistic_regression_weights = False
    logistic_regression_dir = '1/best_model.model'
    in_features = 64*4 + 256*2  # 64 per ogni unet/resnet, 256 per ogni deeplab
    channels = 6 + 7*0 # 1 per ogni rete binaria e 7 per multibinaria
    crop_size = (320,320)

    
    # PATHS management
    paths = Paths(db=db_name, platform=platform)
    paths.dir_logreg = paths.dir_logreg + f'/{logistic_regression_dir}'

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
        
        
        
    

    return predictor.predict()
    


if __name__ == "__main__":
    

    experiment_dict = {
        
        "Feature Fusion":{
            "load_models_dir": {
            "1": "10070/model_best.model",
            "2": "10072/model_best.model",
            "3": "10051/model_best.model",
            "4": "10052/model_best.model",
            "5": "10071/model_best.model",
            "6": "10073/model_best.model",
            "coarse": "10106/model_best.model",
            "4_a": "10090/model_best.model",
            "5_a": "10053/model_best.model"
        },
    
    "models_type_list": {
            "1": "seresunet",
            "2": "seresunet",
            "3": "deeplabv3",
            "4": "deeplabv3",
            "5": "seresunet",
            "6": "seresunet",
            "coarse": "deeplabv3",
            "4_a": "unet",
            "5_a": "deeplabv3",
        },
    "load_dir_metamodel":"1543/model_best.model",
    "stacking_mode":"lastlayer_fusion",
    "convolutional_meta_type":"Onex1StackConv_Unet"
        },
        
        "1x1 Convolution":{
            "load_models_dir": {
            "1": "10070/model_best.model",
            "2": "10072/model_best.model",
            "3": "10051/model_best.model",
            "4": "10052/model_best.model",
            "5": "10071/model_best.model",
            "6": "10073/model_best.model",
            "coarse": "10106/model_best.model",
            "4_a": "10090/model_best.model",
            "5_a": "10053/model_best.model"
        },
    
    "models_type_list": {
            "1": "seresunet",
            "2": "seresunet",
            "3": "deeplabv3",
            "4": "deeplabv3",
            "5": "seresunet",
            "6": "seresunet",
            "coarse": "deeplabv3",
            "4_a": "unet",
            "5_a": "deeplabv3",
        },
    "load_dir_metamodel":"1549/model_best.model",
    "stacking_mode":"convolutional",
    "convolutional_meta_type":"Onex1StackConv_Unet"
        },
        
        "Unet meta-model":{
            "load_models_dir": {
            "1": "10070/model_best.model",
            "2": "10072/model_best.model",
            "3": "10051/model_best.model",
            "4": "10052/model_best.model",
            "5": "10071/model_best.model",
            "6": "10073/model_best.model",
            "coarse": "10106/model_best.model",
            "4_a": "10090/model_best.model",
            "5_a": "10053/model_best.model"
        },
    
    "models_type_list": {
            "1": "seresunet",
            "2": "seresunet",
            "3": "deeplabv3",
            "4": "deeplabv3",
            "5": "seresunet",
            "6": "seresunet",
            "coarse": "deeplabv3",
            "4_a": "unet",
            "5_a": "deeplabv3",
        },
    "load_dir_metamodel":"1550/model_best.model",
    "stacking_mode":"convolutional",
    "convolutional_meta_type":"stack_UNet"
        },
            
        
        
    }
    
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
    
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local"  # local, colab, polimi
    paths = Paths(db=db_name, platform=platform)
    dataset = HDF5Dataset(scale=1, mode='test', db_info=json.load(open(paths.json_file_database)), 
                              hdf5_db_dir=paths.hdf5_db, labels=labels, channels=1,crop_size=(320,320))
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    
    for batch in test_loader:
        mask = batch['mask_gt']
                        
    f = plt.subplot(1, 5, 1)
    plt.rc('axes', titlesize=6, )
    plt.axis("off")
    plt.title("Ground Truth")
    f = plt.imshow(mask.squeeze())

    for i, (val, exp) in enumerate(experiment_dict.items()):
        img = prediction_plot(experiment_dict[val])

    
        plt.subplot(1, 5, i+2)
        plt.rc('axes', titlesize=6)
        plt.axis("off")
        plt.title(val)
        plt.imshow(img)
    
    plt.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig("data/results/slice40_results/t5.png", bbox_inches='tight',pad_inches = 0)
    #plt.show()
    
    
        