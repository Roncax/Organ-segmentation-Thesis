import sys
sys.path.append(r'/home/roncax/Git/organ_segmentation_thesis/') # gradient: Organ-segmentation-Thesis

from OaR_segmentation.utilities.paths import Paths
from OaR_segmentation.training.trainers.CustomTrainer import CustomTrainer
from OaR_segmentation.network_architecture.net_factory import build_net
import argparse



def run_training():

    model = "unet"  # args.network   #seresunet, unet, segnet, deeplabv3
    # args.db  #SegTHOR, StructSeg2019_Task3_Thoracic_OAR
    db_name = "StructSeg2019_Task3_Thoracic_OAR"
    epochs = 1000  # args.epochs
    batch_size = 1  # args.batch_size
    lr = 0.0001  # args.learning_rate
    val = 0.2  # args.validation_size
    patience = 5  # args.patience
    fine_tuning = False  # args.fine_tuning
    feature_extraction = False  # args.feature_extraction
    augmentation = True  # args.augmentation
    train_type = 'fine'  # args.train_type
    deep_supervision = True  # args.deep_supervision #only unet and seresunet
    # args.dropout #deeplav3 builded in, unet and seresunet only (segnet not supported)
    dropout = True
    scale = 1  # args.scale
    # used for multi-channel 3d method (forse problemi con deeplab)
    channels = 1
    multi_loss_weights = [1, 1]  # for composite losses
    deeplabv3_backbone = "mobilenet"  # resnet, drn, mobilenet, xception
    platform = "local"  # local, colab, polimi

    old_classes = 7  # args.old_classes
    paths = Paths(db=db_name, platform=platform)

    labels = {
        "0": "Bg",
        "1": "RightLung",
       "2": "LeftLung",
        "3": "Heart",
        "4": "Trachea",
        "5": "Esophagus",
        "6": "SpinalCord"
    }  # dict_db_parameters["labels"]

    # SegThor
    # "0": "Bg",
    # "2": "Heart",
    # "4": "Aorta",
    # "3": "Trachea",
    # "1": "Esophagus"

    # Structseg
    # "0": "Bg",
    # "1": "RightLung",
    # "2": "LeftLung",
    # "3": "Heart",
    # "4": "Trachea",
    # "5": "Esophagus",
    # "6": "SpinalCord"

    n_classes = len(labels) if len(labels) > 2 else 1

    load_dir_list = {
        "1": "931/model_best.model",
        "2": "931/model_best.model",
        "3": "931/model_best.model",
        "4": "931/model_best.model",
        "5": "931/model_best.model",
        "6": "931/model_best.model",
        "coarse": ""
    }

    # dice, bce, binaryFocal, multiclassFocal, crossentropy, dc_bce
    loss_criteria = {
        "0": "bce",
        "1": "bce",
        "2": "bce",
        "3": "bce",
        "4": "bce",
        "5": "bce",
        "6": "bce",
        "coarse": "crossentropy"
    }

    assert not (
        feature_extraction and fine_tuning), "Finetuning and feature extraction cannot be both active"
    if feature_extraction or fine_tuning:
        assert old_classes > 0, "Old classes needed to be specified"

    if train_type == "coarse":
        paths.set_pretrained_model(load_dir_list["coarse"])

        net = build_net(model=model, n_classes=n_classes, finetuning=fine_tuning, load_dir=paths.dir_pretrained_model,
                        channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                        dropout=dropout, deep_supervision=deep_supervision, backbone=deeplabv3_backbone)

        trainer = CustomTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                batch_size=batch_size, loss_criterion=loss_criteria[
                                    'coarse'], val_percent=val,
                                labels=labels, network=net, deep_supervision=deep_supervision, dropout=dropout,
                                fine_tuning=fine_tuning, feature_extraction=feature_extraction,
                                pretrained_model=load_dir_list["coarse"], lr=lr, patience=patience, epochs=epochs,
                                multi_loss_weights=multi_loss_weights, platform=platform, dataset_name=db_name)

        trainer.initialize()
        trainer.run_training()

    elif train_type == "fine":
        labels_list = filter(lambda x: x != '0', list(labels.keys()))

        for label in labels_list:
            label_dict = {label: labels[label]}
            paths.set_pretrained_model(load_dir_list[label])

            net = build_net(model=model, n_classes=1, finetuning=fine_tuning,
                            load_dir=paths.dir_pretrained_model,
                            channels=channels, old_classes=old_classes, feature_extraction=feature_extraction,
                            dropout=dropout, deep_supervision=deep_supervision, backbone=deeplabv3_backbone)
            print(net.n_classes)

            trainer = CustomTrainer(paths=paths, image_scale=scale, augmentation=augmentation,
                                    batch_size=batch_size, loss_criterion=loss_criteria[
                                        label], val_percent=val,
                                    labels=label_dict, network=net, deep_supervision=deep_supervision, dropout=dropout,
                                    fine_tuning=fine_tuning, feature_extraction=feature_extraction,
                                    pretrained_model=load_dir_list[label], lr=lr, patience=patience, epochs=epochs,
                                    multi_loss_weights=multi_loss_weights, platform=platform, dataset_name=db_name)

            trainer.initialize()
            trainer.run_training()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--network", help="seresunet, unet, segnet, deeplabv3", required=False)
    parser.add_argument("--database_name", "-db",
                        help="Supports: StructSeg2019_Task3_Thoracic_OAR, ...", required=False)
    parser.add_argument("--deterministic", "-det",
                        help="Makes training deterministic, but reduces training speed substantially. "
                             "Deterministic training will make you overfit to some random seed. "
                             "Don't use that.",
                        required=False, default=False)
    parser.add_argument("--epochs", "-e", required=False, default=1000)
    parser.add_argument("--batch_size", "-bs", required=False, default=1)
    parser.add_argument("--learning_rate", "-lr",
                        required=False, default=0.0001)
    parser.add_argument(
        "--scale", help="Downscaling factor of the images", required=False, default=1)
    parser.add_argument("--validation_size", "-val", help="% of the database that is used as validation (0-1)",
                        required=False, default=0.2)
    parser.add_argument("--patience",
                        help="Epochs patience of the early stopping. -1 means no early stopping (save all the epochs)",
                        required=False, default=5)
    parser.add_argument("--fine_tuning", help="Enable finetuning for every net (initialize the net before training)",
                        required=False, default=False)
    parser.add_argument("--feature_extraction", "-fx",
                        help="Enable feature extraction on every net (freeze all except last out layer)",
                        required=False, default=False)
    parser.add_argument("--verbose", "-v", required=False, default=True)
    parser.add_argument("--augmentation", "-aug", required=False, default=True)
    parser.add_argument("--train_type", required=False,
                        default="multiclass", help="multibinary or multiclass")
    parser.add_argument("--deep_supervision", required=False, default=False)
    parser.add_argument("--dropout", required=False, default=False)
    parser.add_argument("--old_classes", required=False, default=0, help="Needed only if fine_tuning or "
                                                                         "feature_extraction, the model need to know "
                                                                         "the last param number of the final layer")
    parser.add_argument("--debug_mode", required=False, default=False,
                        help="Active debug mode if you not want to had permanent effect (e.g. save pth or epoch losses)")
    #args = parser.parse_args()

    run_training()
