import argparse
import os
import os.path as osp

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
import torch

from callbacks import MetricCallback
from datamodule.ell_data import ELL_data
from model.multiregression import MultiRegression
from utils import create_dir
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description="parser option")

# model params
parser.add_argument("--name_model", default="microsoft/deberta-v3-base", type=str)
parser.add_argument("--nb_of_linears", default=0, type=int)
parser.add_argument("--freeze_backbone", default=False, type=bool)
parser.add_argument("--save_pretrained", default="pretrained", type=str)
parser.add_argument("--max_length", default=None, type=int)
parser.add_argument("--layer_norm", default=False, type=bool)
parser.add_argument("--pooling", default="MeanPooling", type=str)

# optimization params
parser.add_argument("--loss", default="SmoothL1Loss", type=str)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--scheduler", default=None, type=str)
parser.add_argument("--T_max", default=1, type=int)
parser.add_argument("--step_size_scheduler", default=1, type=int)
parser.add_argument("--weight_decay", default=0, type=int)
parser.add_argument("--auto_scale_batch_size", default="power")
parser.add_argument("--accumulate_grad_batches", default=None, type=int)
parser.add_argument("--max_epochs", default=-1, type=int)

# dataset params
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--validation_split", default=0.1, type=float)
parser.add_argument("--root", default=osp.join(os.getcwd(), "assets"), type=str)

# trainer params
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--fast_dev_run", default=False, type=bool)
parser.add_argument("--limit_train_batches", default=1.0, type=float)
parser.add_argument("--val_check_interval", default=1.0, type=float)
parser.add_argument("--kaggle", default=False, type=bool)
parser.add_argument("--tune", default=False, type=bool)

# inference params
parser.add_argument("--test", default=False)
parser.add_argument("--ckpt_path", default="exp/weights/charmed-pond-48.ckpt", type=str)

config = parser.parse_args()

if config.kaggle:
    config.root = "/kaggle/input"

config.save_pretrained = osp.join(config.save_pretrained, config.name_model)

if not config.test:

    wandb_tags = [
        config.name_model,
        "freezed_backbone" if config.freeze_backbone else "no_freezed_backbone",
        f"{config.nb_of_linears} linear layers",
        config.loss,
        config.pooling,
    ]

    if config.layer_norm:
        wandb_tags.append("layer_norm")

    wandb_logger = WandbLogger(
        config=config,
        project="ELL",
        entity="clementapa",
        allow_val_change=True,
        log_model="all",
        save_dir=osp.join(os.getcwd(), "exp"),
        tags=wandb_tags,
    )

    save_dir = osp.join(os.getcwd(), "exp", wandb_logger.experiment.name)
    create_dir(save_dir)

    callbacks = [
        ModelCheckpoint(
            monitor="val/mcrmse",
            save_top_k=2,
            mode="min",
            verbose=True,
            filename="epoch={epoch}-step={step}-val_mcrmse{val/mcrmse:.2f}",
            auto_insert_metric_name=False,
            dirpath=osp.join(save_dir, "weights"),
        ),
        LearningRateMonitor(),
        MetricCallback(),
    ]

    if not config.kaggle:
        callbacks += [RichProgressBar()]

    trainer = Trainer(
        logger=wandb_logger,
        gpus=config.gpu,
        auto_scale_batch_size=config.auto_scale_batch_size,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_checkpointing=True,
        fast_dev_run=config.fast_dev_run,
        limit_train_batches=config.limit_train_batches,
        val_check_interval=config.val_check_interval,
        accumulate_grad_batches=config.accumulate_grad_batches,
        default_root_dir=save_dir,
        max_epochs=config.max_epochs,
    )

    model = MultiRegression(config)
    dataset_module = ELL_data(config)

    if config.tune:
        trainer.tune(model, datamodule=dataset_module)
    trainer.fit(model, datamodule=dataset_module)

else:
    model = MultiRegression(config)
    dataset_module = ELL_data(config)

    trainer = Trainer(
        gpus=config.gpu,
        auto_scale_batch_size=config.auto_scale_batch_size,
        default_root_dir=osp.join(os.getcwd(), "exp"),
    )

    preds = trainer.predict(
        model, datamodule=dataset_module, ckpt_path=config.ckpt_path
    )

    raw_predictions = torch.cat(preds, axis=0)
    y_pred = raw_predictions.detach().cpu().numpy()
    y_pred = np.clip(y_pred, 1, 5)
    text_id = dataset_module.predict_set.df["text_id"]

    output_df = pd.DataFrame(
        {
            "text_id": {},
            "cohesion": {},
            "syntax": {},
            "vocabulary": {},
            "phraseology": {},
            "grammar": {},
            "conventions": {},
        }
    )

    output_df["text_id"] = text_id

    for i, label in enumerate(model.labels):
        output_df[label] = y_pred[:, i]

    output_df.to_csv(
        f"submission.csv",
        index=False,
    )

    print("submission.csv written !")
