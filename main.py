import os
import os.path as osp

import torch.nn as nn
from easydict import EasyDict as edict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint, RichProgressBar)
from pytorch_lightning.loggers import WandbLogger

from model.multiregression import MultiRegression
from datamodule.ell_data import ELL_data    


os.environ["TOKENIZERS_PARALLELISM"] = "false"

name_model = "bert-base-uncased"

config = {
    "lr": 0.001,
    "loss": nn.MSELoss(reduction='mean'),
    "name_model": name_model,
    "batch_size": 6,
    "num_workers": 4,
    "fast_dev_run":False,
    "limit_train_batches": 1.0,
    "val_check_interval": 1.0,
    "validation_split": 0.1,
    "root": osp.join(os.getcwd(), "assets"),
    "freeze_backbone": False
}

config = edict(config)

wandb_logger = WandbLogger(
    config=config,
    project="ELL",
    entity="clementapa",
    allow_val_change=True,
    save_dir=osp.join(os.getcwd())    
)
callbacks = [
            ModelCheckpoint(            
                monitor="val/loss", 
                dirpath= osp.join(os.getcwd(), "weights"), #'/kaggle/working/', 
                filename="best-model",
                mode="min",
                verbose=True
            ),
            RichProgressBar(),
            LearningRateMonitor(),
    ]

trainer = Trainer(
    logger=wandb_logger,
    gpus=0,
    auto_scale_batch_size="power",
    callbacks=callbacks,
    log_every_n_steps=1,
    enable_checkpointing=True,
    fast_dev_run=config.fast_dev_run,
    limit_train_batches=config.limit_train_batches,
    val_check_interval=config.val_check_interval,
    )

model = MultiRegression(config)
dataset_module = ELL_data(config)
trainer.fit(model, dataset_module)
