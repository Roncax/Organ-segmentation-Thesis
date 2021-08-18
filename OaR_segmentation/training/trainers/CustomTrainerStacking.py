import json
from numpy.lib.function_base import place

from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from OaR_segmentation.db_loaders.HDF5DatasetStacking import HDF5Dataset_stacking
from OaR_segmentation.training.loss_factory import build_loss
from OaR_segmentation.training.trainers.NetworkTrainer import NetworkTrainer
from OaR_segmentation.evaluation import eval


class CustomTrainerStacking(NetworkTrainer):
    def __init__(self, paths, image_scale, augmentation, batch_size
                 , loss_criterion, val_percent, labels, network, deep_supervision, pretrained_model, dropout,
                 feature_extraction, fine_tuning, lr, epochs, patience, dataset_name, multi_loss_weights,platform, used_output_models, deterministic=False,
                 fp16=True):
        super(CustomTrainerStacking, self).__init__(deterministic, fp16)

        self.paths = paths
        self.network = network
        self.deep_supervision = deep_supervision
        self.pretrained_model = pretrained_model
        self.dropout = dropout
        self.feature_extraction = feature_extraction
        self.fine_tuning = fine_tuning
        self.max_num_epochs = epochs
        self.patience = patience
        self.dataset_name = dataset_name

        self.output_folder = None
        self.loss_criterion = loss_criterion
        self.multi_loss_weights = multi_loss_weights
        self.class_weights = None
        self.dataset_directory = paths.dir_database
        self.lr = lr
        self.platform = platform
        self.used_output_models = used_output_models

        self.img_scale = image_scale
        self.labels = labels
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.val_percent = val_percent

        self.all_loss_custom_val = []
        self.experiment_number = None

    def set_experiment_number(self):
        dict_db_parameters = json.load(open(self.paths.json_stacking_experiments_results))
        dict_db_parameters[f"experiments_{self.platform}"] += 1
        self.experiment_number = dict_db_parameters[f"experiments_{self.platform}"]
        json.dump(dict_db_parameters, open(self.paths.json_stacking_experiments_results, "w"))
        self.paths.set_experiment_stacking_number(self.experiment_number)
        self.output_folder = self.paths.dir_checkpoint

    def initialize(self, training=True):
        super(CustomTrainerStacking, self).initialize(training)

        self.class_weights = [int(x) / 100 for x in json.load(open(self.paths.json_file_database))["weights"].values()]

        self.set_experiment_number()
        self.loss = build_loss(loss_criterion=self.loss_criterion, class_weights=self.class_weights,
                               bce_dc_weights=self.multi_loss_weights, deep_supervision=self.deep_supervision)
        self.initialize_optimizer_and_scheduler()
        self.load_dataset()

        self.tensorboard_setup()

        self.print_to_log_file(f"Experiment {self.experiment_number}")
        self.print_to_log_file(f"Organ(s) {list(self.labels.values())}")
        self.print_to_log_file(f"Model {self.network.name}")
        self.json_log_save()

        self.was_initialized = True

    def initialize_optimizer_and_scheduler(self):
        self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.lr, weight_decay=1e-8, momentum=0.9)
        #self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=0.9)
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, mode='min', patience=2)

    def validate(self, *args, **kwargs):
        loss_custom_val = eval.eval_train(self.network, self.val_gen, self.device)
        self.all_loss_custom_val.append(loss_custom_val)
        self.print_to_log_file(f"My custom validation loss is {self.all_loss_custom_val[-1]}")

        self.update_json_train()
        self.update_tsboard()

    def load_dataset(self):
        # DATASET split train/val
        self.dataset = HDF5Dataset_stacking(scale=self.img_scale, hdf5_db_dir=self.paths.hdf5_stacking,
                                   labels=self.labels, augmentation=self.augmentation, channels=self.network.n_channels)

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
            'Custom_validation': self.all_loss_custom_val[-1]
        }

        self.writer.add_scalar("Loss/train", self.all_tr_losses[-1], self.epoch)
        self.writer.add_scalar(tag='Loss/validation', scalar_value=self.all_val_losses[-1], global_step=self.epoch)
        self.writer.add_scalar(tag='Loss/custom_validation', scalar_value=self.all_loss_custom_val[-1],
                               global_step=self.epoch)
        self.writer.add_scalars(main_tag='Losses', tag_scalar_dict=temp_dict, global_step=self.epoch)
        self.writer.add_scalars(main_tag='MA', tag_scalar_dict={"Train": self.train_loss_MA,
                                                                "Validation": self.val_eval_criterion_MA},
                                global_step=self.epoch)

        #self.writer.add_images('example_images', self.example_imgs, self.epoch)

        for tag, value in self.network.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), self.epoch)

            try:
                self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.epoch)
            except AttributeError:
                continue

        self.writer.flush()

    def json_log_save(self):
        temp_dict = {
            self.experiment_number: {
                "dataset":self.dataset_name,
                "organ": list(self.labels.values()),
                "model": self.network.name,
                "used_output_models": self.used_output_models,
                "max_num_epochs": self.max_num_epochs,
                "epoch_results": {},
                "batch_size": self.batch_size,
                "learning_rate": self.lr,
                "validation_size": self.val_percent,
                "patience": self.patience,
                "feature_extraction": self.feature_extraction,
                "deep_supervision": self.deep_supervision,
                "dropout": self.dropout,
                "fine_tuning": self.fine_tuning,
                "augmentation": self.augmentation,
                "loss_criteria": self.loss_criterion,
                "loss_weights": self.multi_loss_weights,
                "pretrained_model": self.pretrained_model,
                "class_weights": self.class_weights,
                "train_loss_MA_alpha": self.train_loss_MA_alpha,
                "train_loss_MA_eps": self.train_loss_MA_eps,
                "val_eval_criterion_alpha": self.val_eval_criterion_alpha
            }
        }
        temp = json.load(open(self.paths.json_file_train_results))
        temp.update(temp_dict)
        json.dump(temp, open(self.paths.json_file_train_results, "w"))

    def update_json_train(self):
        temp_dict = {self.epoch:
                         {"validation_loss": float(self.all_val_losses[-1]),
                          "avg_train_loss": float(self.all_tr_losses[-1]),
                          "loss_custom_val": float(self.all_loss_custom_val[-1]),
                          "val_eval_criterion_MA": float(self.val_eval_criterion_MA),
                          "train_loss_MA": float(self.train_loss_MA)
                          }
                     }

        dict_results = json.load(open(self.paths.json_file_train_results))
        dict_results[str(self.experiment_number)]["epoch_results"].update(temp_dict)
        json.dump(dict_results, open(self.paths.json_file_train_results, "w"))
