import sys

from albumentations import augmentations
# gradient: Organ-segmentation-Thesis
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset
from OaR_segmentation.utilities.data_vis import visualize, visualize_test
from OaR_segmentation.training.trainers.CustomTrainerStacking import CustomTrainerStacking
from OaR_segmentation.network_architecture.net_factory import build_net
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np
import h5py
import json
from OaR_segmentation.utilities.paths import Paths



def create_combined_dataset(scale, nets,  paths, labels):
    dataset = HDF5Dataset(scale=scale, mode='test', db_info=json.load(open(paths.json_file_database)), 
                          hdf5_db_dir=paths.hdf5_db, channels=1, labels=labels)
    
    test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

    with h5py.File(paths.hdf5_stacking, 'w') as db:
        with tqdm(total=len(dataset), unit='img') as pbar:
            for i, batch in enumerate(test_loader):
                imgs = batch['dict_organs']
                mask_gt = batch['mask_gt']
                
                mask_gt = mask_gt.squeeze().cpu().numpy()
                db.create_dataset(f"{i}/gt", data=mask_gt)

                for organ in nets.keys():
                    nets[organ].eval()
                    img = imgs[organ].to(device="cuda", dtype=torch.float32)

                    with torch.no_grad():
                        output = nets[organ](img)

                    probs = output
                    probs = torch.sigmoid(probs) # todo log_softmax, raw logits, log_sigmoid, softmax
                    full_mask = probs.squeeze().squeeze().cpu().numpy()

                    # TESTING
                    # img_t = img.clone().detach().squeeze().cpu().numpy()
                    # full_mask_thresholded = full_mask > mask_threshold
                    # print(organ)
                    # visualize(image=full_mask, mask=img_t, additional_1=full_mask_thresholded, additional_2=mask_gt ,file_name=f"{i}_{organ}")
                    # if(i==0 and organ == "3"):
                    #     print("hey")
                    # res = np.array(full_mask).astype(np.bool)

                    db.create_dataset(f"{i}/{organ}", data=full_mask)

                # update the pbar by number of imgs in batch
                pbar.update(img.shape[0])
                

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
    loss_criterion = 'crossentropy'
    lr = 0.02
    patience = 5
    deep_supervision = False
    dropout = False
    fine_tuning = False
    batch_size = 1
    scale = 1
    augmentations = False
    feature_extraction = False
    epochs = 500
    validation_size = 0.2
    multi_loss_weights=[1, 1]
    channels = 6
    find_lr = False

    nets = {}
    for label in labels.keys():
        paths.set_pretrained_model(load_dir_list[label])
        paths.set_train_stacking_results()

        nets[label] = build_net(model=models[label], n_classes=1, channels=channels,
                                load_inference=True, load_dir=paths.dir_pretrained_model)

    if db_prediction_creation:
        create_combined_dataset(nets=nets, scale=scale, paths=paths, labels=labels)


    net = build_net(model='stack_UNet', n_classes=n_classes, channels=channels, load_inference=False)

    trainer = CustomTrainerStacking(paths=paths, image_scale=scale, augmentation=augmentations,
                                    batch_size=batch_size, loss_criterion=loss_criterion, val_percent=validation_size,
                                    labels=labels, network=net, deep_supervision=deep_supervision, dropout=dropout,
                                    fine_tuning=fine_tuning, feature_extraction=feature_extraction,
                                    pretrained_model='', lr=lr, patience=patience, epochs=epochs,
                                    multi_loss_weights=multi_loss_weights, platform=platform, 
                                    used_output_models=models, dataset_name=db_name)

    trainer.initialize()
    if find_lr:
        trainer.find_lr(num_iters=2000)
    else:
        trainer.run_training()
