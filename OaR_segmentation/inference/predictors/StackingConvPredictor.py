import json
from albumentations.augmentations.functional import crop
import h5py
import numpy as np
import torch
from torch._C import device
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import pandas as pd

from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.utilities.data_vis import visualize
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset

class StackingConvPredictor(Predictor):
    def __init__(self, scale, mask_threshold,  paths, labels, n_classes, logistic_regression_weights, crop_size):
        super(StackingConvPredictor, self).__init__(scale = scale, mask_threshold = mask_threshold,
                                                    paths=paths, labels=labels, n_classes=n_classes
                                                    , logistic_regression_weights=logistic_regression_weights, crop_size=crop_size)
        self.nets = None
        self.meta_net = None
        self.channels = None
        
        
    def initialize(self, load_dir_metamodel, channels, load_models_dir, models_type_list, meta_net_model):
        super(StackingConvPredictor, self).initialize()
        self.channels = channels
        self.nets = self.initialize_multinets(load_models_dir=load_models_dir, models_type_list= models_type_list)
        self.meta_net = self.initialize_metamodel(load_dir_metamodel, self.n_classes, meta_net_model)
    
    
    def initialize_metamodel(self, load_dir_metamodel, n_classes, meta_net_model):
        self.paths.set_pretrained_model(load_dir_metamodel)
        net =  build_net(model=meta_net_model, n_classes=n_classes, channels=self.channels, load_inference=True,load_dir=self.paths.dir_pretrained_model)
        
        # Show the tensor weights of the 1x1 convolution
        # d = {}
        # col_temp=[]
        # for i, c in enumerate(self.nets.keys()):
        #     if c == "coarse":
        #         col_temp.extend([f"c_{x}" for x in range(7)])
        #     else:
        #         col_temp.append(c)
                
        #     d[i]=np.append(net.outc.conv.weight[i].detach().cpu().numpy().squeeze(),net.outc.conv.bias[i].item())
            
        # col_temp.append('bias')
        # df = pd.DataFrame.from_dict(d, orient='index', columns=col_temp)
        # print(df)
        
        return net
        
        
    def initialize_multinets(self, load_models_dir, models_type_list):
        nets = {}
        for label in self.labels.keys():
            self.paths.set_pretrained_model(load_models_dir[label])

            nets[label] = build_net(model=models_type_list[label], n_classes=1 if label != 'coarse' else self.n_classes, 
                                         channels=1, load_inference=True, 
                                         load_dir=self.paths.dir_pretrained_model)
        return nets
    

    def predict(self):
        super(StackingConvPredictor, self).predict()
        
        dataset = HDF5Dataset(scale=self.scale, mode='test', 
                              db_info=json.load(open(self.paths.json_file_database)), 
                              hdf5_db_dir=self.paths.hdf5_db, labels=self.labels, channels=1,crop_size=self.crop_size)
        test_loader = DataLoader(dataset=dataset, batch_size=1, 
                                 shuffle=True, num_workers=8, pin_memory=True)

        loss = nn.CrossEntropyLoss()
        l=0
 
        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['dict_organs']
                    mask = batch['mask_gt']
                    id = batch['id']
                    
                    final_array_prediction = None

                    # For every binary net, do a single inference and concatenate all
                    for organ in self.nets.keys():
                        self.nets[organ].eval()
                        img = imgs[organ].to(device="cuda", dtype=torch.float32)
                        with torch.no_grad():
                            output = self.nets[organ](img)
                            output = F.sigmoid(output) #! probabile problema con 1x1 convolution
                            output = output.to(device="cuda", dtype=torch.float32)

                        if final_array_prediction is None:
                            final_array_prediction = output
                        else:
                            final_array_prediction = torch.cat((final_array_prediction, output), dim=1)
     

                    final_array_prediction = final_array_prediction.to(device="cuda", dtype=torch.float32)
                    
                    # Feed the concatenated outputs to the metamodel net
                    self.meta_net.eval()                    
                    with torch.no_grad():
                        stacking_output = self.meta_net(final_array_prediction)
                    
                    if self.logistic_regression_weights:
                        stacking_output = self.apply_logistic_weights(stacking_output)
                    
                    #for CE in test
                    mask = mask.squeeze(dim=1)
                    mask = mask.to(dtype=torch.long, device='cuda:0')
                    l += loss(stacking_output, mask)
                    
                    
                    probs = stacking_output    
                    probs = F.softmax(probs, dim=1)
                    probs = probs.squeeze().cpu().numpy()

                    probs = self.combine_predictions(output_masks=probs)
                    
                    # probs = stacking_output 
                    
                    # probs = torch.squeeze(probs)
                    # row_exclude =0
                    # probs = torch.cat((probs[:row_exclude],probs[row_exclude+1:]))
                    # probs = torch.unsqueeze(probs, dim=0)
                    
                    # probs = F.softmax(probs, dim=1) #todo test sigmoid
                    # probs = probs.squeeze().cpu().detach().numpy()

                    # probs = self.combine_predictions(output_masks=probs) #np.delete(probs, 0, 0)

                    # TESTING
                    # mask = mask.squeeze().cpu().numpy()
                    # test = final_array_prediction.squeeze().cpu().numpy()
                    # test = self.combine_predictions(output_masks=test, background=False)
                    # visualize(image=probs, mask=mask, additional_1=test, additional_2=test, file_name='temp_img')
                    
                    db.create_dataset(id[0], data=probs)    #  add the calculated image in the hdf5 results file
                    pbar.update(img.shape[0])   # update the pbar by number of imgs in batch

        print(f"CE test loss: {l/len(test_loader)}")
        

    