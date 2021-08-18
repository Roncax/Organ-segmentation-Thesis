import sys

sys.path.append(r'/home/roncax/Git/Pytorch-UNet/') # /content/gdrive/MyDrive/Colab/Thesis_OaR_Segmentation/

from OaR_segmentation.utilities.paths import Paths
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.training.custom_trainer_stacking import CustomTrainerStacking
from OaR_segmentation.utilities.data_vis import visualize, visualize_test

from OaR_segmentation.datasets.hdf5Dataset import HDF5Dataset




def create_combined_dataset(scale, mask_threshold, nets,  paths, labels):
    dataset = HDF5Dataset(scale=scale, mode='stacking', db_info=json.load(open(paths.json_file_database)), hdf5_db_dir=paths.hdf5_db, channels=1, labels=labels)
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    with h5py.File(paths.hdf5_stacking, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for i, batch in enumerate(test_loader):
                imgs = batch['img_dict']
                mask_gt = batch['mask_gt']


                mask_gt = mask_gt.to(device="cuda", dtype=torch.float32)
                mask_gt=mask_gt.squeeze().cpu().numpy()

                db.create_dataset(f"{i}/gt", data=mask_gt)
            
                for organ in nets.keys():
                    nets[organ].eval()
                    img = imgs[organ].to(device="cuda", dtype=torch.float32)

                    #logits
                    with torch.no_grad():
                        output = nets[organ](img)
                        
                    probs = output                    
                    #probs = F.log_softmax(probs)
                    probs = torch.sigmoid(probs)
                    probs = probs.squeeze(0)
                    full_mask = probs.squeeze().cpu().numpy() 
                    
                    # img_t = img.clone().detach().squeeze().cpu().numpy()
                    # full_mask_thresholded = full_mask > mask_threshold
                    # print(organ)
                    # visualize(image=full_mask, mask=img_t, additional_1=full_mask_thresholded, additional_2=mask_gt ,file_name=f"{i}_{organ}")
                    # if(i==0 and organ == "3"):
                    #     print("hey")
                    # res = np.array(full_mask).astype(np.bool)
                    
                    db.create_dataset(f"{i}/{organ}", data=full_mask)


                pbar.update(img.shape[0])  # update the pbar by number of imgs in batch

        
    
def stacking_training(paths, labels, platform, used_output_models, dataset_name):
    loss_criterion = 'crossentropy'
    lr = 1e-4
    net = build_net(model='stack_UNet', n_classes=7, channels=6, load_inference=False)
    
    
    trainer = CustomTrainerStacking( paths=paths, image_scale=1, augmentation=False,
                            batch_size=1, loss_criterion=loss_criterion, val_percent=0.2,
                            labels=labels, network=net, deep_supervision=False, dropout=False,
                            fine_tuning=False, feature_extraction=False,
                            pretrained_model='', lr=lr, patience=5, epochs=1000,
                            multi_loss_weights=[1,1], platform=platform, used_output_models=used_output_models, dataset_name= dataset_name)

    trainer.initialize()
    trainer.run_training()   
    
    
    
if __name__=="__main__":
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    platform = "local" #local, colab, polimi
    db_prediction_creation = False


    load_dir_list = {
        "1": "1048/model_best.model",
        "2": "1049/model_best.model",
        "3": "1051/model_best.model",
        "4": "1052/model_best.model",
        "5": "1053/model_best.model",
        "6": "1054/model_best.model",
        "coarse": "931/model_best.model"
    }
    models = {"1": "unet",
              "2": "unet",
              "3": "unet",
              "4": "unet",
              "5": "unet",
              "6": "unet",
              "coarse": "stack_unet"
              }
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception

    labels = {
        "0": "Bg",
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }
    n_classes = len(labels) if len(labels) > 2 else 1
    scale = 1
    mask_threshold = 0.5
    channels = 1
    lr = 1e-5

    paths = Paths(db=db_name, platform=platform)

    labels.pop("0")  # don't want to predict also the background
    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])
        paths.set_train_stacking_results()

        nets[label] = build_net(model=models[label], n_classes=1, channels=channels, load_inference=True, load_dir=paths.dir_pretrained_model)
    
    if db_prediction_creation:
        create_combined_dataset(nets=nets,scale=scale, paths=paths, labels=labels, mask_threshold=mask_threshold )
    

    stacking_training(paths=paths, labels=labels, platform=platform, used_output_models = load_dir_list, dataset_name = db_name)
