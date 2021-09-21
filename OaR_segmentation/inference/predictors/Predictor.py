from torch import nn
from torch.nn import functional
from OaR_segmentation.network_architecture.logistic_regression_stacking import LogisticRegression
from abc import abstractmethod
import json
import logging
import h5py
from tqdm import tqdm
import numpy as np
import torch

from OaR_segmentation.utilities.build_volume import grayscale2rgb_mask
from OaR_segmentation.utilities.data_vis import prediction_plot, volume2gif, plot_single_result, boxplot_plotly
from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.evaluation import metrics
from OaR_segmentation.evaluation.metrics import ConfusionMatrix


class Predictor(object):
    def __init__(self, scale, mask_threshold, paths, labels, n_classes, logistic_regression_weights):
        self.scale = scale
        self.mask_threshold = mask_threshold
        self.paths = paths
        self.labels = labels
        self.n_classes = n_classes
        self.logistic_regression_weights = logistic_regression_weights
    
    @abstractmethod
    def initialize(self):
        pass
        
    @abstractmethod
    def predict(self):
        pass
    
        
    def combine_predictions(self, output_masks, threshold = None):
        """Combine the output masks in one single dimension and threshold it. 
        The returned matrix has value in range (1, shape(output_mask)[0])

        Args:
            output_masks (Cxnxn numpy matrix): Output nets matrix

        Returns:
            (nxn) numpy matrix: A combination of all output mask in the first dimension of the matrix
        """
        if threshold is not None:
            self.mask_threshold = threshold
            
        matrix_shape = np.shape(output_masks[0])
        combination_matrix = np.zeros(shape=matrix_shape)
        
        output_masks[not np.argmax(output_masks)] = 0
        output_masks[output_masks >= self.mask_threshold] = 1
        output_masks[output_masks < self.mask_threshold] = 0

        for i in range(np.shape(output_masks)[0]):
            combination_matrix[output_masks[i,:,:] == 1] = i+1 #on single dimension - single image
            #full_output_mask[i, full_output_mask[i, :, :] == 1] = i+1 # on multiple dimension - multiple images 

        return combination_matrix
    
    
    def compute_save_metrics(self, metrics_list, db_name, colormap, experiment_num, dict_test_info, sample_gif_name="volume_46", gif_viz = False):
        """Compute, save and plot the specified metrics for every volume in the hdf5_results dir

        Args:
            metrics_list (list): metric list
            db_name (string): name of db (path saving purpose)
            colormap (dict): association between organ and rgb colors
            experiment_num (int): number of inference experiment
            dict_test_info (dict): testing info
            sample_gif_name (str, optional): name of gif (paths name purpose). Defaults to "volume_46".
            gif_viz (bool, optional): to save or not the example gif. Defaults to False.
        """        
        
        results = {}
        self.paths.set_plots_folder(experiment_number = experiment_num)

        with h5py.File(self.paths.hdf5_results, 'r') as db:
            with h5py.File(self.paths.hdf5_db, 'r') as db_train:
                with tqdm(total=len(db[f'{db_name}/test'].keys()), unit='volume') as pbar:
                    for volume in db[f'{db_name}/test'].keys():
                        results[volume] = {}
                        vol = []
                        pred_vol = None
                        gt_vol = None

                        for slice in sorted(db[f'{db_name}/test/{volume}/image'].keys(),
                                            key=lambda x: int(x.split("_")[1])):
                            slice_pred_mask = db[f'{db_name}/test/{volume}/image/{slice}'][()]
                            slice_gt_mask = db_train[f'{db_name}/test/{volume}/mask/{slice}'][()]
                            slice_test_img = db_train[f'{db_name}/test/{volume}/image/{slice}'][()]

                            # GIF management
                            if volume == sample_gif_name and gif_viz:
                                msk = grayscale2rgb_mask(colormap=colormap, labels=self.labels, mask=slice_pred_mask)
                                gt = grayscale2rgb_mask(colormap=colormap, labels=self.labels, mask=slice_gt_mask)
                                plot = prediction_plot(img=slice_test_img, mask=msk, ground_truth=gt)
                                vol.append(plot)

                            # Adding slices to volumes
                            slice_pred_mask = np.expand_dims(slice_pred_mask, axis=2)
                            slice_gt_mask = np.expand_dims(slice_gt_mask, axis=2)
                            
                            if pred_vol is None and gt_vol is None:
                                pred_vol = slice_pred_mask
                                gt_vol = slice_pred_mask
                            else:
                                assert pred_vol is not None and gt_vol is not None, "Uncongruent slices"
                                pred_vol = np.append(pred_vol, slice_pred_mask, axis=2).astype(dtype=int)                                
                                gt_vol = np.append(gt_vol, slice_gt_mask, axis=2).astype(dtype=int)

                        if volume == sample_gif_name and gif_viz:
                            volume2gif(volume=vol, target_folder=self.paths.dir_plots,
                                    out_name=f"example({volume})_inference({experiment_num})")

                        # metrics computation with confusion matrix and store results
                        for l in self.labels.keys():
                            pred_vol_cp = np.zeros(pred_vol.shape)
                            gt_vol_cp = np.zeros(gt_vol.shape)
                            pred_vol_cp[pred_vol == int(l)] = 1
                            gt_vol_cp[gt_vol == int(l)] = 1
                            cm = ConfusionMatrix(test=pred_vol_cp, reference=gt_vol_cp)
                            results[volume][self.labels[l]] = cm

                        pbar.update(1)

                    # plot results
                    for m in metrics_list:
                        results_dict = self.save_results(results=results, path_json=self.paths.json_file_inference_results, met=m,
                                                        experiment_num=experiment_num, test_info=dict_test_info, labels=self.labels, 
                                                        path_settings=self.paths.json_experiments_settings)

                        boxplot_plotly(score=results_dict, type=m, path=self.paths.dir_plots, exp_num=experiment_num, colors=colormap)


    # calculate and save all the metrics
    def save_results(self, results, path_json, path_settings, met, labels, experiment_num, test_info):
        dict_results = json.load(open(path_json))
        dict_results[experiment_num] = test_info
        dict_results[experiment_num][met] = {}
        score = {}
        for organ in labels:
            score[labels[organ]] = []

        print(f"\nCalculating {met} now")
        with tqdm(total=len(results.keys()), unit='volume') as pbar:
            for patient in results:
                for organ in results[patient]:
                    score[organ].append(metrics.ALL_METRICS[met](confusion_matrix=results[patient][organ]))
                pbar.update(1)

        for organ in score:
            d = {
                "data": score[organ],
                "avg": np.average(score[organ]),
                "min": np.min(score[organ]),
                "max": np.max(score[organ]),
                "25_quantile": np.quantile(score[organ], q=0.25),
                "75_quantile": np.quantile(score[organ], q=0.75),
                "median": np.median(score[organ])
            }
            dict_results[experiment_num][met][organ] = d
            
        dict_experiments_settings = json.load(open(path_settings))
        dict_experiments_settings["inference_experiment"] = experiment_num


        json.dump(dict_results, open(path_json, "w"))
        json.dump(dict_experiments_settings, open(path_settings, "w"))
        return score


    def apply_logistic_weights(self, img):
        
        net = LogisticRegression(input_size=512*512, n_classes=self.n_classes)
        net = net.to(device='cuda')
        ckpt = torch.load(self.paths.dir_logreg, map_location='cuda')
        net.load_state_dict(ckpt['state_dict'])
        weights = net.linear.weight.clone()
        
        img = img*(torch.nn.functional.functional.normalize(weights))
        #img = torch.tensor [C, H, W]
        # caricare modello logistico
        #moltiplicare modello logistico per ogni organo (da normalizzare prima)
        return img