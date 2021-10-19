import json
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.utilities.data_vis import visualize
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset

class LastLayerPredictor(Predictor):
    def __init__(self, scale, mask_threshold,  paths, labels, n_classes, logistic_regression_weights, in_features):
        super(LastLayerPredictor, self).__init__(scale = scale, mask_threshold = mask_threshold,  
                                                 paths=paths, labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        self.nets = None
        self.meta_net = None
        self.channels = None
        self.in_features = in_features
        
        
    def initialize(self, load_dir_metamodel, channels, load_models_dir, models_type_list):
        super(LastLayerPredictor, self).initialize()
        self.channels = channels
        self.nets = self.initialize_multinets(load_models_dir=load_models_dir, models_type_list= models_type_list)
        self.meta_net = self.initialize_metamodel(load_dir_metamodel, self.n_classes)
    
    
    def initialize_metamodel(self, load_dir_metamodel, n_classes):
        self.paths.set_pretrained_model(load_dir_metamodel)
        
        return build_net(model='fusion_net', n_classes=n_classes, channels=1, load_inference=True,
                         load_dir=self.paths.dir_pretrained_model, nets=self.nets, n_labels=len(self.nets), in_features=self.in_features)
        
        
    def initialize_multinets(self, load_models_dir, models_type_list):
           # Restore all nets
        nets = {}
        for label in self.labels.keys():
            if 'coarse' in label:
                n_c = self.n_classes
            else:
                n_c=1
                
        
            self.paths.set_pretrained_model(load_models_dir[label])

            nets[label] = build_net(model=models_type_list[label], n_classes=n_c, 
                                         channels=self.channels, load_inference=True, 
                                         load_dir=self.paths.dir_pretrained_model, lastlayer_fusion=True)
        return nets
    

    def predict(self):
        super(LastLayerPredictor, self).predict()
        
        dataset = HDF5Dataset(scale=self.scale, mode='test', 
                              db_info=json.load(open(self.paths.json_file_database)), 
                              hdf5_db_dir=self.paths.hdf5_db, labels=self.labels, channels=1)
        test_loader = DataLoader(dataset=dataset, batch_size=1, 
                                 shuffle=True, num_workers=8, pin_memory=True)

        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['dict_organs']
                    mask = batch['mask_gt']
                    id = batch['id']
                    
                    for organ in self.nets.keys():
                        self.nets[organ].eval()
                        imgs[organ] = imgs[organ].to(device="cuda", dtype=torch.float32)
                    
                    # Feed the concatenated outputs to the metamodel net
                    self.meta_net.eval()                    
                    with torch.no_grad():
                        stacking_output = self.meta_net(imgs)
                    
                    
                    if self.logistic_regression_weights:
                        stacking_output = self.apply_logistic_weights(stacking_output)
                    
                    probs = stacking_output    
                    probs = F.softmax(probs, dim=1)
                    probs = probs.squeeze().cpu().numpy()

                    probs = self.combine_predictions(output_masks=np.delete(probs, 0, 0))

                    # TESTING
                    #mask = mask.squeeze().cpu().numpy()
                    #visualize(image=probs, mask=probs, additional_1=mask, additional_2=mask, file_name='temp_img/x.png')
                    
                    db.create_dataset(id[0], data=probs)    #  add the calculated image in the hdf5 results file
                    pbar.update(n=1)   # update the pbar by number of imgs in batch


    