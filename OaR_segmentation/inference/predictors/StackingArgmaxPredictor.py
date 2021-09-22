import numpy as np
from OaR_segmentation.utilities.data_vis import visualize
import json
import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from OaR_segmentation.network_architecture.net_factory import build_net
from OaR_segmentation.inference.predictors.Predictor import Predictor
from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset


class StackingArgmaxPredictor(Predictor):
    def __init__(self, scale, mask_threshold,  paths, labels, n_classes, logistic_regression_weights):
        super(StackingArgmaxPredictor, self).__init__(scale = scale, mask_threshold = mask_threshold,  
                                                      paths=paths, labels=labels, n_classes=n_classes, logistic_regression_weights=logistic_regression_weights)
        self.nets = None
        self.channels = None
        
        
    def initialize(self, channels, load_models_dir, models_type_list):
        super(StackingArgmaxPredictor, self).initialize()
        self.channels = channels
        self.nets = self.initialize_multinets(load_models_dir=load_models_dir, models_type_list= models_type_list)
    

    def initialize_multinets(self, load_models_dir, models_type_list):
        nets = {}
        for label in self.labels.keys():
            self.paths.set_pretrained_model(load_models_dir[label])

            nets[label] = build_net(model=models_type_list[label], n_classes=1, 
                                    channels=self.channels, load_inference=True,
                                    load_dir=self.paths.dir_pretrained_model)
        
        return nets

        
    def predict(self):
        super(StackingArgmaxPredictor, self).predict()
        
        dataset = HDF5Dataset(scale=self.scale, mode='test', db_info=json.load(open(self.paths.json_file_database)), 
                              hdf5_db_dir=self.paths.hdf5_db, labels=self.labels, channels=self.channels)
        test_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

        with h5py.File(self.paths.hdf5_results, 'w') as db:
            with tqdm(total=len(dataset), unit='img') as pbar:
                for batch in test_loader:
                    imgs = batch['dict_organs']
                    id = batch['id']
                    final_array_prediction = None

                    for organ in self.nets.keys():
                        self.nets[organ].eval()
                        img = imgs[organ].to(device="cuda", dtype=torch.float32)

                        with torch.no_grad():
                            output = self.nets[organ](img)
                            output = torch.sigmoid(output)
                        
                        if final_array_prediction is None:
                            final_array_prediction = output
                        else:
                            final_array_prediction = torch.cat((final_array_prediction, output), dim=1)

                    if self.logistic_regression_weights:
                        final_array_prediction = self.apply_logistic_weights(final_array_prediction)
                    probs = final_array_prediction
                    #probs = torch.sigmoid(probs)
                    full_mask = probs.squeeze().cpu().detach().numpy()
                    comb_img = self.combine_predictions(output_masks=full_mask)
                    
                    
                    # TESTING
                    # real_img = batch['image']
                    # real_img = real_img.squeeze().cpu().numpy()
                    # mask = batch['mask']
                    # mask = mask.squeeze().cpu().numpy()
                    # raw_output = final_array_prediction.squeeze().cpu().numpy()
                    # raw_output = self.combine_predictions(output_masks=raw_output)
                    # visualize(image=real_img, mask=mask, additional_1=raw_output, additional_2=comb_img)

                    db.create_dataset(id[0], data=comb_img) # add the calcualted image in the hdf5 results file
                    pbar.update(img.shape[0])   # update the pbar by number of imgs in batch