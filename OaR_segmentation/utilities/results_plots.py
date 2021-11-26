import json
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')


def create_pandas_df(json_metrics, metric, net_list, organ_list, metric_list):

    for organ, t_lst in json_metrics[metric].items():
        data = np.array(t_lst['data'])
        
        net_name =  json_metrics['stacking_mode'] if json_metrics['stacking_mode']!='convolutional' else json_metrics['convolutional_meta_type'] 

        switcher={
            'none': "DeepLabV3",
            'argmax': "ArgMax",
            'lastlayer_fusion': "Feature Fusion",
            'Onex1StackConv_Unet': "1x1 convolution",
            'stack_UNet': "Unet meta-model",
            'unet':"Unet",
            'deeplabv3':"DeepLabV3",
            'seresunet':"SE-ResUnet",
        }
        
        
        net_name_t = switcher[net_name]
        
        for d in data:
            if math.isnan(d):
                d=0
                if metric=='Hausdorff Distance 95':
                    d=120
            metric_list.append(d)
            organ_list.append(organ)
            net_list.append(net_name_t)
        
    
    return net_list, organ_list, metric_list


def exp_boxplots(dict_results, experiments, metric, t):

    organ_list = []
    net_list=[]
    metric_list=[]

    for e in experiments:
        net_list, organ_list, metric_list = create_pandas_df(dict_results[e], metric, net_list, organ_list, metric_list)


    full_db = {'net':net_list,
               'organ':organ_list,
                metric:metric_list}
    
    pd.DataFrame(full_db)
    #sns.set(rc={"figure.figsize":(30, 15)}, font_scale = 2)
    sns.boxplot(data = full_db, y = metric, x = 'organ', hue = 'net')
    plt.title(metric)
    plt.savefig(f"data/results/boxplots/{t}_{metric}", transparent=True)
    #plt.show()
    plt.close()

def mean_and_csv_metrics(dict_results, experiments, metric):
    tmean_dict = {}
    
    for e in experiments:
        mean_values=[]
        
        net_name = dict_results[e]['stacking_mode'] if dict_results[e]['stacking_mode']!='convolutional' else dict_results[e]['convolutional_meta_type'] 


        switcher={
            'none': "Unet",
            'argmax': "ArgMax",
            'lastlayer_fusion': "Feature Fusion",
            'Onex1StackConv_Unet': "1x1 convolution",
            'stack_UNet': "Unet meta-model",
            'unet':"Unet",
            'deeplabv3':"DeepLabV3",
            'seresunet':"SE-ResUnet",
        }
        
        net_name_t = switcher[net_name]

        for organ, t_lst in dict_results[e][metric].items():
            data = np.array(t_lst['data'])
            data = np.nan_to_num(data, copy=True, nan=120.0 if metric == 'Hausdorff Distance 95' else 0, posinf=None, neginf=None)
            mean = np.mean(data)
            mean_values.append(mean)
            
        tmean_dict[net_name_t] = round(np.nanmean(mean_values), 4)
        
    
    return tmean_dict

if __name__ == '__main__':
    t="t3_esophagus"
    experiments = ["82","106" ,"92", "93","94"]
    metrics_list =['Dice',  'Hausdorff Distance 95', 'Precision', 'Recall']
    file_path="data/results/inference_results.json"
    dict_results = json.load(open(file_path))

    d_tot={}
    for metric in metrics_list:
        exp_boxplots(dict_results,experiments, metric, t)
        d_tot[metric]= mean_and_csv_metrics(dict_results, experiments, metric)

    df = pd.DataFrame(d_tot)
    df.to_csv(f'data/results/boxplots/mean_results_csv_temp.csv')

    
    #p = [4.3, 73.8, 3.3, 18.6] #[44.5, 35, 16.3, 0.9, 1.3, 2]
    #l = ["Esophagus", "Heart", "Trachea", "Aorta"] #["RightLung", "LeftLung", "Heart", "Trachea", "Esophagus", "SpinalCord"]
    
    #sns.barplot(x=p, y=l)
    #plt.show()
    
   # d_p = {"Background":[97.6, 98.9],
    #       "Organs":[2.4, 1.1],
    #       "Datasets":["StructSeg", "SegTHOR"]}

    #sns.barplot(x="Datasets", y="Background", data=d_p, color="red")

   # sns.barplot(x="Datasets", y="Organs", data=d_p, color="blue")

   # plt.show()
