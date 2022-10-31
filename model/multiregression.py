import os.path as osp

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from loss import MCRMSELoss
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from model.model import Model


class MultiRegression(pl.LightningModule):
    def __init__(self, config):
        super(MultiRegression, self).__init__()

        self.config = config

        self.lr = config.encoder_lr
        self.batch_size = config.batch_size

        if config.loss == "MCRMSELoss":
            self.loss = MCRMSELoss()
        elif config.loss == "SmoothL1Loss":
            self.loss = nn.SmoothL1Loss()
        elif config.loss == "HuberLoss":
            self.loss = nn.HuberLoss()
        else:
            raise NotImplementedError(f"{config.loss} not implemented !")

        self.model = Model(
            config.name_model,
            config.nb_of_linears,
            config.layer_norm,
            config.pooling,
            config.last_layer_reinitialization,
            config.gradient_checkpointing,
            osp.join(config.save_pretrained, "model"),
        )

        # freeze backbone for fine tuned
        if config.freeze_backbone:
            print("Freezing features extractor")
            for param in self.model.features_extractor.base_model.parameters():
                param.requires_grad = False

        self.labels = [
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ]

    def configure_optimizers(self):
        """defines model optimizer"""
        if not self.config.layer_wise_lr_decay:
            out_dict = {}

            model_parameters = self.get_optimizer_encoder_decoder_params(self.model)

            out_dict["optimizer"] = optim.AdamW(
                model_parameters,
                lr=self.lr,
                weight_decay=self.config.weight_decay,
            )

            if self.config.scheduler != None:
                if self.config.scheduler == "CosineAnnealingLR":
                    out_dict["lr_scheduler"] = optim.lr_scheduler.CosineAnnealingLR(
                        out_dict["optimizer"], self.config.T_max
                    )
                elif self.config.scheduler == "StepLR":
                    out_dict["lr_scheduler"] = optim.lr_scheduler.StepLR(
                        out_dict["optimizer"], self.config.step_size_scheduler
                    )
                elif self.config.scheduler == "ReduceLROnPlateau":
                    out_dict["lr_scheduler"] = {
                        "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                            out_dict["optimizer"], mode="min", patience=3
                        ),
                        "monitor": "train/loss",
                    }
                elif self.config.scheduler == "linear":
                    out_dict["lr_scheduler"] = get_linear_schedule_with_warmup(
                        out_dict["optimizer"],
                        num_warmup_steps=0,
                        num_training_steps=self.config.num_train_steps,
                    )
                elif self.config.scheduler == "cosine":
                    out_dict["lr_scheduler"] = get_cosine_schedule_with_warmup(
                        out_dict["optimizer"],
                        num_warmup_steps=0,
                        num_training_steps=self.config.num_train_steps,
                    )
                else:
                    raise NotImplementedError(
                        f"{self.config.scheduler} scheduler not supported"
                    )
        else:
            out_dict = {}
            grouped_optimizer_params = self.get_optimizer_llrd_grouped_parameters(
                self.model, self.lr, self.config.weight_decay, self.config.LLDR
            )
            out_dict["optimizer"] = optim.AdamW(
                grouped_optimizer_params,
                lr=self.lr,
                eps=self.config.adam_epsilon,
            )
            if self.config.scheduler == "linear":
                out_dict["lr_scheduler"] = get_linear_schedule_with_warmup(
                    out_dict["optimizer"],
                    num_warmup_steps=0,
                    num_training_steps=self.config.num_train_steps,
                )
            else:
                out_dict["lr_scheduler"] = get_cosine_schedule_with_warmup(
                    out_dict["optimizer"],
                    num_warmup_steps=0,
                    num_training_steps=self.config.num_train_steps,
                )

        return out_dict

    def get_optimizer_encoder_decoder_params(self, model):
        """
        https://www.kaggle.com/code/rhtsingh/guide-to-huggingface-schedulers-differential-lrs/notebook
        """
        # differential learning rate and weight decay
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {
                "params": [
                    p
                    for n, p in model.features_extractor.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "lr": self.config.encoder_lr,
                "weight_decay_rate": 0.01,
            },
            {
                "params": [
                    p
                    for n, p in model.features_extractor.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "lr": self.config.encoder_lr,
                "weight_decay_rate": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "features_extractor" not in n
                ],
                "lr": self.config.decoder_lr,
                "weight_decay_rate": 0.01,
            },
        ]
        return optimizer_parameters

    def get_optimizer_llrd_grouped_parameters(
        self, model, learning_rate, weight_decay, layerwise_learning_rate_decay
    ):
        """
        https://www.kaggle.com/code/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning?scriptVersionId=67176591&cellId=26
        """
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "cls" in n
                    or "pooler" in n
                    or "linears" in n
                    or "layer_norm" in n
                ],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
        # initialize lrs for every layer
        num_layers = model.features_extractor.config.num_hidden_layers
        layers = [model.features_extractor.base_model.embeddings] + list(
            model.features_extractor.base_model.encoder.layer
        )
        layers.reverse()
        lr = learning_rate
        for layer in layers:
            lr *= layerwise_learning_rate_decay
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        return optimizer_grouped_parameters

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, preds = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train/loss", loss.mean())

        return {"loss": loss.mean(), "preds": preds.detach()}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, preds = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val/loss", loss.mean())

        # Let's return preds to use it in a custom callback
        return {"preds": preds}

    def predict_step(self, batch, batch_idx):
        inputs = batch
        preds = self(inputs)
        return preds

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        inputs, targets = batch
        preds = self(inputs)

        loss = self.loss(preds, targets["labels"])

        return loss, preds
