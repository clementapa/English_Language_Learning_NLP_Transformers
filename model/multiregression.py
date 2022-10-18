from model.model import Model
import torch
import torch.optim as optim
import pytorch_lightning as pl
import os.path as osp
from loss import MCRMSELoss
import torch.nn as nn


class MultiRegression(pl.LightningModule):
    def __init__(self, config):
        super(MultiRegression, self).__init__()

        self.config = config

        self.lr = config.lr
        self.batch_size = config.batch_size

        if config.loss == "MCRMSELoss":
            self.loss = MCRMSELoss()
        elif config.loss == "SmoothL1Loss":
            self.loss = nn.SmoothL1Loss()
        else:
            raise NotImplementedError(f"{config.loss} not implemented !")

        self.model = Model(
            config.name_model,
            config.nb_of_linears,
            config.layer_norm,
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
        out_dict = {}
        out_dict["optimizer"] = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.config.weight_decay
        )
        if self.config.scheduler != None:
            if self.config.scheduler == "CosineAnnealingLR":
                out_dict["scheduler"] = optim.lr_scheduler.CosineAnnealingLR(
                    out_dict["optimizer"], self.config.T_max
                )
            elif self.config.scheduler == "StepLR":
                out_dict["scheduler"] = optim.lr_scheduler.StepLR(
                    out_dict["optimizer"], self.config.step_size_scheduler
                )
            elif self.config.scheduler == "ReduceLROnPlateau":
                out_dict["lr_scheduler"] = {
                    "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                        out_dict["optimizer"], mode="min", patience=3
                    ),
                    "monitor": "train/loss",
                }
            else:
                raise NotImplementedError(
                    f"{self.config.scheduler} scheduler not supported"
                )
        return out_dict

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
