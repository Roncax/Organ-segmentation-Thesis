import json
import sys
import pandas as pd
import numpy as np

sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/')


def create_pandas_metrics(json_metrics, metrics_list, labels):
    temp_dict = {}
    for metric in metrics_list:
        organ_list = []
        for organ, t_lst in json_metrics[metric].items():
            data = np.array(t_lst['data'])
            mean = round(np.mean(data), 4)
            std_dev = round(np.std(data), 3)
            wr_data = f'{mean} ± {std_dev}'
            organ_list.append(wr_data)

        temp_dict[metric] = organ_list
    
    return pd.DataFrame(temp_dict, index=list(labels.values()))



if __name__ == '__main__':
    
    file_path="data/results/inference_results.json"
    experiment = "36"
    metrics_list =['Dice', 'Precision', 'Recall',  'Avg. Surface Distance', 'Hausdorff Distance 95'] 

    labels = {
        "1": "RightLung",
        "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }

    dict_results = json.load(open(file_path))
    db = create_pandas_metrics(dict_results[experiment], metrics_list, labels=labels)
    db.to_csv(f'data/results/plots/{experiment}/to_csv.csv')
    print(db)