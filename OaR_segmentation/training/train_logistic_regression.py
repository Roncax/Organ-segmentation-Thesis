import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from OaR_segmentation.db_loaders.HDF5DatasetStacking import HDF5DatasetStacking
import torch
import os
from OaR_segmentation.db_loaders.HDF5logreg import HDF5DatasetLogReg
from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.network_architecture.logistic_regression_stacking import LogisticRegression
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')

from  matplotlib import pyplot as plt

def viz_weights(weights, path, epoch):
    
    p = plt.figure(figsize=(30, 2.5))
    for i in range(n_classes):
        scale = np.max(np.abs(weights[i]))
        p = plt.subplot(1, n_classes, i + 1)
        p = plt.imshow(weights[i].reshape(512, 512),
                    cmap=plt.cm.RdBu, vmin=0, vmax=scale)
        p = plt.axis('off')
        p = plt.title('Class %i' % i)
    #plt.show()
    os.makedirs(f"{path}", exist_ok=True)
    plt.savefig(path + f"/epoch_{epoch}.png")
    plt.close()

if __name__ == "__main__":

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
    db_prediction_creation = True
    n_classes = 6  
    scale = 1
    paths = Paths(db=db_name, platform=platform)
    loss_criterion = 'crossentropy' # dice, focal, crossentropy, dc_ce
    lr = 0.02
    patience = 20
    batch_size = 1
    scale = 1
    epochs = 500
    validation_size = 0.2
    
    
    name = "logreg_experiments"
    dict_db_parameters = json.load(open(paths.json_experiments_settings))
    dict_db_parameters[name] += 1
    experiment_number = dict_db_parameters[name]
    json.dump(dict_db_parameters, open(paths.json_experiments_settings, "w"))
    paths.set_experiment_logreg_number(experiment_number)
    
    
    net = LogisticRegression(input_size=512*512, n_classes=n_classes)
    net = net.to(device='cuda')
    
    dataset = HDF5DatasetLogReg(scale=scale, hdf5_db_dir=paths.hdf5_db, labels=labels, db_info=json.load(open(paths.json_file_database)), mode='train')

    n_val = int(len(dataset) * validation_size)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    tr_gen = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_gen = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    history = [] # for recording epoch-wise results
    
    best_eval_loss = -1
    current_eval_loss = -1
    patience_count = 0
    
    for epoch in range(epochs):
        
        # TRAINING
        net.train()
        with trange(len(tr_gen), unit='batch', leave=False) as tbar: 
            for i, batch in enumerate(tr_gen):
                
                for organ in labels.keys():
                    image = batch['masks'][organ]
                    if torch.sum(image) <1:
                        continue
                    label = torch.tensor(data = int(organ)-1).view(1)
                    optimizer.zero_grad()
                    loss = net.training_step(img=image, organ=label)
                    loss.backward()
                    optimizer.step()
                
                tbar.update(list(image.shape)[0])
        
        viz_weights(weights = net.linear.weight.clone().detach().cpu().numpy(), path=paths.dir_checkpoint, epoch=epoch)

        # VALIDATION
        outputs = []
        net.eval()
        with trange(len(val_gen), unit='batch', leave=False) as tbar: 
            for i, batch in enumerate(val_gen):
                for organ in labels.keys():
                    image = batch['masks'][organ]
                    if torch.sum(image) <1:
                        continue
                    label = torch.tensor(data = int(organ)-1).view(1)
                    output = net.validation_step(img=image, organ=label)
                    outputs.append(output)
                tbar.update(list(image.shape)[0])
                
        result = net.validation_epoch_end(outputs)
        net.epoch_end(epoch, result)
        history.append(result)

        # SAVE BEST MODEL and EARLY STOPPING
        current_eval_loss = result['val_loss']
        if epoch == 0:
            best_eval_loss = current_eval_loss
                        
            torch.save({'state_dict': net.state_dict()},f'{paths.dir_checkpoint}/best_model.model')

        else:
            if current_eval_loss < best_eval_loss:
                torch.save({'state_dict': net.state_dict()},f'{paths.dir_checkpoint}/best_model.model')
                best_eval_loss = current_eval_loss
                patience_count = 0
            else:
                patience_count += 1
        
        if patience_count >= patience:
            print("My patience ended")
            break
                
                
        
    

