import os
import pathlib
from argparse import ArgumentParser
import pytorch_lightning as pl
from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule
from module import DenseNetPredictSens
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
import datetime
import wandb
import matplotlib.pyplot as plt 

def main(args):
    number = int(args.runs[-1])
    pl.seed_everything(2000 + number)  

    wandb_logger = WandbLogger(
        project=args.project,
        name="acc" + str(args.acceleration) + "x_" + args.runs,
        notes=args.notes,
        entity="entityname",
        save_dir=args.log_path,
    )
    # ------------
    # data
    # ------------
    # this creates a k-space mask for transforming input data
    if int(args.acceleration) < 25:
        mask = create_mask_for_mask_type("equispaced", [0.04], [int(args.acceleration)])
    else:
        mask = create_mask_for_mask_type("equispaced", [0.01], [int(args.acceleration)])
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=True, use_pads=True)
    val_transform = VarNetDataTransform(mask_func=mask, use_seed=True, use_pads=True)
    test_transform = VarNetDataTransform(mask_func=mask, use_seed=True, use_pads=True)
    # ptl data module - this handles data loaders
    data_module = FastMriDataModule(
        data_path=args.data_path,
        anns_path=args.anns_path,
        maps_path=args.maps_path,
        test_path=args.data_path / "multicoil_mytest",
        challenge="multicoil",
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        sample_rate=0.99,
        batch_size=args.batch_size,
        num_workers=0,
        num_classes=args.num_classes,
        mode='fastmri-binary',
        use_dataset_cache_file=False
    )
    # ------------
    # model
    # ------------
    model = DensebigSenseAbs(
        sens_pools=4,  # number of pooling layers for sense est. U-Net
        sens_chans=8,  # number of top-level channels for sense est. U-Net
        lr=0.001,  # Adam learning rate
        lr_step_size=10,  # epoch at which to decrease learning rate
        lr_gamma=0.2,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight regularization strength)
        make_transforms= True
    
    )
    # .load_from_checkpoint(path)

    wandb_logger.watch(model)
    wandb.config.update(args)
    # ------------
    # trainer
    # ------------
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=args.log_path / "checkpoints",
            save_top_k=True,
            verbose=True,
            save_on_train_epoch_end= True
        )
    ]

    
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="ddp",
        gpus=-1,
        check_val_every_n_epoch=1,
        default_root_dir=args.log_path,  # directory for logs and checkpoints
        max_epochs=70,
        callbacks=callbacks,
        log_every_n_steps=1,

    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("dirs.yaml")

    # set defaults based on optional directory config
    data_path = fetch_dir("brain_path", path_config)
    maps_path = fetch_dir("maps_path", path_config)
    anns_path = fetch_dir("anns_path", path_config)
    ct = datetime.datetime.now()
    timestring = "modelname" + str(ct)
    default_root_dir = fetch_dir("log_path", path_config) / timestring
    parser.add_argument(
        "--log_path",
        default=default_root_dir,
        help="logging path",
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type= int,
        help="batch_size",
    )

    parser.add_argument(
        "--data_path",
        default=data_path,
        help="data path",
    )

    parser.add_argument(
        "--maps_path",
        default=maps_path,
        help="sens maps path",
    )

    parser.add_argument(
        "--anns_path",
        default=anns_path,
        help="annotations path",
    )
    # add wandb arguments
    parser.add_argument(
        "--project",
        default="project name",
        type=str,
        help="wandb project name",
    )

    parser.add_argument(
        "--runs",
        default="run0",
        type=str,
        help="wandb # of runs",
    )

    parser.add_argument(
        "--notes",
        default=" ",
        type=str,
        help="helper notes for the run",
    )
    # client arguments
    parser.add_argument(
        "--mode",
        default="train",
        choices=("train", "test"),
        type=str,
        help="Operation mode",
    )

    # data transform params
    parser.add_argument(
        "--acceleration",
        default=1,
        type=int,
        help="Acceleration rates to use for masks",
    )

    parser.add_argument(
        "--datamode",
        default=" ",
        type=str,
        help="Annotation mode",
    )
    parser.add_argument(
        "--num_classes",
        default=2,
        type=int,
        help="number of classes",
    )

    args = parser.parse_args()

    return args


def train():
    args = build_args()

    main(args)


if __name__ == "__main__":
    train()
