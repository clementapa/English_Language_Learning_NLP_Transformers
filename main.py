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

from callbacks import MetricCallback
from datamodule.ell_data import ELL_data
from model.multiregression import MultiRegression

parser = argparse.ArgumentParser(description="parser option")
parser.add_argument("--name_model", default="microsoft/deberta-v3-base")
parser.add_argument("--lr", default=0.001)
parser.add_argument("--batch_size", default=6, type=int)
parser.add_argument("--num_workers", default=4)
parser.add_argument("--fast_dev_run", default=False)
parser.add_argument("--limit_train_batches", default=1.0)
parser.add_argument("--val_check_interval", default=1.0)
parser.add_argument("--validation_split", default=0.1)
parser.add_argument("--root", default=osp.join(os.getcwd(), "assets"))
parser.add_argument("--freeze_backbone", default=False)
parser.add_argument("--max_length", default=512)
parser.add_argument("--gpu", default=0)
parser.add_argument("--auto_scale_batch_size", default="power")
parser.add_argument("--accumulate_grad_batches", default=None)

config = parser.parse_args()

wandb_logger = WandbLogger(
    config=config,
    project="ELL",
    entity="clementapa",
    allow_val_change=True,
    save_dir=osp.join(os.getcwd()),
)
callbacks = [
    ModelCheckpoint(
        monitor="val/loss",
        dirpath=osp.join(os.getcwd(), "weights"),  #'/kaggle/working/',
        filename="best-model",
        mode="min",
        verbose=True,
    ),
    RichProgressBar(),
    LearningRateMonitor(),
    MetricCallback(),
]

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
)

model = MultiRegression(config)
dataset_module = ELL_data(config)
trainer.fit(model, dataset_module)
