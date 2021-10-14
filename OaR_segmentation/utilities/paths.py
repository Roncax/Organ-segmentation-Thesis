import os


class Paths:
    
    """Simple data class to store all useful paths
    """    
    
    def __init__(self, db, platform) -> None:
        super().__init__()

        # Variable
        self.db_name = db
        self.dir_pretrained_model = None
        self.dir_checkpoint = None


        if platform == "local":
            self.dir_root = '/home/roncax/Git/organ_segmentation_thesis'
        elif platform == "gradient":
            self.dir_root = '/notebooks/Organ-segmentation-Thesis'
        elif platform == "polimi":
            self.dir_root = '/notebooks/Organ-segmentation-Thesis'
            
        # Directories
        self.dir_database = f'{self.dir_root}/data/datasets/{self.db_name}'
        self.dir_raw_db = f'{self.dir_database}/raw_data'
        self.json_file_database = f'{self.dir_database}/{self.db_name}.json'
        self.json_experiments_settings = f'{self.dir_root}/data/results/experiments_settings.json'

        self.dir_plots = f'{self.dir_root}/data/results/plots'
        self.dir_stacking = f"{self.dir_root}/data/checkpoints_stacking"
        self.dir_segmentation = f"{self.dir_root}/data/checkpoints_segmentation"
        self.dir_logreg = f"{self.dir_root}/data/checkpoints_logreg"

        # HDF5
        self.hdf5_db = f"{self.dir_database}/{self.db_name}.hdf5"
        self.hdf5_results = f"{self.dir_database}/{self.db_name}_predictions.hdf5"
        self.hdf5_results = f"{self.dir_database}/{self.db_name}_stacking.hdf5"
        self.hdf5_stacking = f"{self.dir_database}/temp_stacking_db.hdf5"

        # Json
        self.json_file_train_results = f"{self.dir_root}/data/results/train_results_{platform}.json"
        self.json_file_inference_results = f"{self.dir_root}/data/results/inference_results.json"


    def set_experiment_number(self, n):
        self.dir_checkpoint = f'{self.dir_segmentation}/{n}'
        os.makedirs(self.dir_checkpoint, exist_ok=True)
        
    def set_experiment_logreg_number(self, n):
        self.dir_checkpoint = f'{self.dir_logreg}/{n}'
        os.makedirs(self.dir_checkpoint, exist_ok=True)
        
    def set_plots_folder(self, experiment_number):
        self.dir_plots = self.dir_plots+f'/{experiment_number}'
        os.makedirs(self.dir_plots, exist_ok=True)

    def set_pretrained_model(self, dir):
        self.dir_pretrained_model = f'{self.dir_segmentation}/{dir}'

