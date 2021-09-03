import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')

from OaR_segmentation.db_loaders.HDF5DatasetStacking import HDF5DatasetStacking
import torch

from OaR_segmentation.db_loaders.HDF5logreg import HDF5DatasetLogReg
from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.network_architecture.logistic_regression_stacking import LogisticRegression
from torch.utils.data import DataLoader, random_split
from tqdm import trange
import json
import numpy as np
import matplotlib.pyplot as plt


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
    channels = 1
    paths = Paths(db=db_name, platform=platform)
    loss_criterion = 'crossentropy' # dice, focal, crossentropy, dc_ce
    lr = 0.02
    patience = 5
    batch_size = 1
    scale = 1
    epochs = 500
    validation_size = 0.2
    telegram = False


    net = LogisticRegression(input_size=512*512, n_classes=n_classes)
    net = net.to(device='cuda')
    
    dataset = HDF5DatasetLogReg(scale=1, hdf5_db_dir=paths.hdf5_db, labels=labels, db_info=json.load(open(paths.json_file_database)), mode='train')

    n_val = int(len(dataset) * validation_size)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    tr_gen = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    val_gen = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    history = [] # for recording epoch-wise results
    
    for epoch in range(epochs):
        
        with trange(len(tr_gen), unit='batch', leave=False) as tbar: 
            for i, batch in enumerate(tr_gen):
                
                for organ in labels.keys():
                    image = batch['masks'][organ]
                    if torch.sum(image) <1:
                        continue
                    label = torch.tensor(data = int(organ)-1)
                    optimizer.zero_grad()
                    loss = net.training_step(img=image, organ=label.view(1))
                    loss.backward()
                    optimizer.step()
                
                if not i%50:
                    scale = np.max(np.abs(net.lineat.))

                    p = plt.figure(figsize=(25, 2.5));

                    for i in range(n_classes):
                        p = plt.subplot(1, n_classes, i + 1)
                        p = plt.imshow(net.coef_[i].reshape(512, 512),
                                    cmap=plt.cm.RdBu, vmin=-scale, vmax=scale);
                        p = plt.axis('off')
                        p = plt.title('Class %i' % i);
                tbar.update(1)
                
        outputs = []
        with trange(len(val_gen), unit='batch', leave=False) as tbar: 
            for i, batch in enumerate(val_gen):
                for organ in labels.keys():
                    image = batch['masks'][organ]
                    label = torch.tensor(data = int(organ)-1)
                    label = label.view(1)
                    output = net.validation_step(img=image, organ=label)
                    outputs.append(output)
                
                result = net.validation_epoch_end(outputs)
                net.epoch_end(epoch, result)
                history.append(result)
                
                print(history)
        
    

