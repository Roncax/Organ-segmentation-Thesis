import json
from numpy.lib.function_base import select

from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from OaR_segmentation.db_loaders.HDF5Dataset import HDF5Dataset
from OaR_segmentation.db_loaders.HDF5DatasetStacking import HDF5DatasetStacking
from OaR_segmentation.db_loaders.HDF5lastlayer import HDF5lastlayer


from OaR_segmentation.training.loss_factory import build_loss
from OaR_segmentation.training.trainers.NetworkTrainer import NetworkTrainer
from OaR_segmentation.evaluation import evaluation

import telegram_send

class ConvolutionTrainer(NetworkTrainer):
    def __init__(self, paths, image_scale, augmentation, batch_size, loss_criterion, val_percent, labels, network, 
                 lr, epochs, patience, multi_loss_weights, platform, dataset_name, optimizer_type, lastlayer_fusion=False, telegram=False, deep_supervision = False, stacking = False, deterministic=False,
                 fp16=True):
        super(ConvolutionTrainer, self).__init__(deterministic=deterministic, fp16=fp16)

        self.paths = paths
        self.network = network
        self.deep_supervision = deep_supervision
        self.max_num_epochs = epochs
        self.patience = patience
        self.dataset_name = dataset_name

        self.output_folder = paths.dir_plots
        self.loss_criterion = loss_criterion
        self.multi_loss_weights = multi_loss_weights
        self.class_weights = None
        self.dataset_directory = paths.dir_database
        self.lr = lr
        self.platform = platform
        self.telegram=telegram

        self.img_scale = image_scale
        self.labels = labels
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.val_percent = val_percent
        self.stacking = stacking
        self.lastlayer_fusion = lastlayer_fusion

        self.info_dict = None
        self.experiment_number = None

        self.optimizer_type = optimizer_type
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 4
        self.initial_lr = 3e-4
        self.weight_decay = 1e-8

    def set_experiment_number(self):
        name = "stacking_experiments" if self.stacking else f"experiments_{self.platform}"
        
        dict_db_parameters = json.load(open(self.paths.json_experiments_settings))
        dict_db_parameters[name] += 1
        self.experiment_number = dict_db_parameters[name]
        json.dump(dict_db_parameters, open(self.paths.json_experiments_settings, "w"))
        if self.stacking:
            self.paths.set_experiment_stacking_number(self.experiment_number)
        else:
            self.paths.set_experiment_number(self.experiment_number)
        self.output_folder = self.paths.dir_checkpoint

    def initialize(self, training=True):
        super(ConvolutionTrainer, self).initialize(training)

        self.class_weights = [int(
            x) / 100 for x in json.load(open(self.paths.json_file_database))["weights"].values()]

        self.set_experiment_number()
        self.loss = build_loss(loss_criterion=self.loss_criterion, class_weights=self.class_weights,
                               ce_dc_weights=self.multi_loss_weights, deep_supervision=self.deep_supervision, n_classes=self.network.n_classes)
        self.initialize_optimizer_and_scheduler()
        self.load_dataset()

        self.tensorboard_setup()

        self.print_to_log_file(f"Experiment {self.experiment_number}")
        self.print_to_log_file(f"Organ(s) {list(self.labels.values())}")
        self.print_to_log_file(f"Model {self.network.name}")

        self.was_initialized = True

    def initialize_optimizer_and_scheduler(self):
        if self.optimizer_type == "rmsprop":
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        elif self.optimizer_type == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), self.lr, weight_decay=self.weight_decay, amsgrad=True)

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                                                 patience=self.lr_scheduler_patience,
                                                                 verbose=True, threshold=self.lr_scheduler_eps,
                                                                 threshold_mode="abs")

    def validate(self):
        loss_custom_val = evaluation.eval_train(
            self.network, self.val_gen, self.device)
        self.all_val_eval_metrics.append(loss_custom_val)
        self.print_to_log_file(
            f"my real validation loss is {self.all_val_eval_metrics[-1]}")

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        self.update_json_train()
        self.update_tsboard()
        if self.telegram:
            self.send_telegram_update(f"Epoch {self.epoch} ended, validation loss {round(self.all_val_eval_metrics[-1], 4)}")
            if not continue_training:
                self.send_telegram_update(f"Patience ended, training done!")

        return continue_training


    def load_dataset(self):
        # DATASET split train/val
        if self.stacking:
            self.dataset = HDF5DatasetStacking(scale=self.img_scale, hdf5_db_dir=self.paths.hdf5_stacking,
                                    labels=self.labels, augmentation=self.augmentation, 
                                    channels=self.network.n_channels)
        if self.lastlayer_fusion:
            self.dataset = HDF5lastlayer(scale=self.img_scale, mode='train',
                                    db_info=json.load(open(self.paths.json_file_database)), hdf5_db_dir=self.paths.hdf5_db,
                                    labels=self.labels, augmentation=self.augmentation, channels=self.network.n_channels, lastlayer_fusion=self.lastlayer_fusion)

        else:
            self.dataset = HDF5Dataset(scale=self.img_scale, mode='train',
                                    db_info=json.load(open(self.paths.json_file_database)), hdf5_db_dir=self.paths.hdf5_db,
                                    labels=self.labels, augmentation=self.augmentation, channels=self.network.n_channels, lastlayer_fusion=self.lastlayer_fusion)

        n_val = int(len(self.dataset) * self.val_percent)
        n_train = len(self.dataset) - n_val
        train, val = random_split(self.dataset, [n_train, n_val])
        self.tr_gen = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                                 drop_last=True)
        self.val_gen = DataLoader(val, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                  drop_last=True)


    def tensorboard_setup(self):
        self.writer = SummaryWriter(log_dir=f'{self.paths.dir_checkpoint}')


    def update_tsboard(self):

        temp_dict = {
            "Train": self.all_tr_losses[-1],
            "Validation": self.all_val_losses[-1],
            'Real_validation': self.all_val_eval_metrics[-1]
        }

        self.writer.add_scalar(
            "Loss/train", self.all_tr_losses[-1], self.epoch)
        self.writer.add_scalar(
            tag='Loss/validation', scalar_value=self.all_val_losses[-1], global_step=self.epoch)
        self.writer.add_scalar(tag='Loss/real_validation', scalar_value=self.all_val_eval_metrics[-1],
                               global_step=self.epoch)
        self.writer.add_scalars(
            main_tag='Losses', tag_scalar_dict=temp_dict, global_step=self.epoch)
        self.writer.add_scalars(main_tag='MA', tag_scalar_dict={"Train": self.train_loss_MA,
                                                                "Validation": self.val_eval_criterion_MA},
                                global_step=self.epoch)

        #self.writer.add_images('example_images', self.example_imgs, self.epoch)

        for tag, value in self.network.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram(
                'weights/' + tag, value.data.cpu().numpy(), self.epoch)

            try:
                self.writer.add_histogram(
                    'grads/' + tag, value.grad.data.cpu().numpy(), self.epoch)
            except AttributeError:
                continue

        self.writer.flush()


    def json_log_save(self):
        temp = json.load(open(self.paths.json_file_train_results))
        temp.update(self.info_dict)
        json.dump(temp, open(self.paths.json_file_train_results, "w"))
        
    


    def update_json_train(self):
        temp_dict = {self.epoch:
                     {"validation_loss": float(self.all_val_losses[-1]),
                      "avg_train_loss": float(self.all_tr_losses[-1]),
                      "loss_real_val": float(self.all_val_eval_metrics[-1]),
                      "val_eval_criterion_MA": float(self.val_eval_criterion_MA),
                      "train_loss_MA": float(self.train_loss_MA)
                      }
                     }

        dict_results = json.load(open(self.paths.json_file_train_results))
        dict_results[str(self.experiment_number)]["epoch_results"].update(temp_dict)
        json.dump(dict_results, open(self.paths.json_file_train_results, "w"))
        
    

    def setup_info_dict(self, pretrained_model='NA', dropout = 'NA', feature_extraction = 'NA', fine_tuning = 'NA', used_output_models = None):
        self.info_dict = {
            self.experiment_number: {
                "dataset": self.dataset_name,
                "organ": list(self.labels.values()),
                "model": self.network.name,
                "max_num_epochs": self.max_num_epochs,
                "batch_size": self.batch_size,
                "initial_learning_rate": self.lr,
                "validation_size": self.val_percent,
                "patience": self.patience,
                "feature_extraction": feature_extraction,
                "deep_supervision": self.deep_supervision,
                "dropout": dropout,
                "fine_tuning": fine_tuning,
                "augmentation": self.augmentation,
                "loss_criteria": self.loss_criterion,
                "loss_weights": self.multi_loss_weights,
                "pretrained_model": pretrained_model,
                "class_weights": self.class_weights,
                "train_loss_MA_alpha": self.train_loss_MA_alpha,
                "train_loss_MA_eps": self.train_loss_MA_eps,
                "val_eval_criterion_alpha": self.val_eval_criterion_alpha,
                "lr_scheduler_eps": self.lr_scheduler_eps,
                "lr_scheduler_patience": self.lr_scheduler_patience,
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "optimizer": self.optimizer_type,
                "epoch_results": {},
            }
        }
        
        if self.stacking: self.info_dict["used_output_models"]=used_output_models

        self.json_log_save()

    def send_telegram_update(self, msg):
        pre  = "Stacking" if self.stacking else "Segmentation"
        telegram_send.send(messages=[f"{pre} {self.experiment_number}: \n{msg}"])
