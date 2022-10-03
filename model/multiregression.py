from model.model import Model
import torch
from torch.optim import Adam
import torch.nn as nn
import pytorch_lightning as pl 

class MultiRegression(pl.LightningModule):
    def __init__(self, config):
        super(MultiRegression, self).__init__()

        self.loss = config.loss
        self.lr = config.lr  
        
        self.model = Model(config.name_model)
        
        # freeze backbone for fine tuned
        for name, param in self.model.named_parameters():
            if 'cls_list' not in name: # classifier layers
                param.requires_grad = False
        
        self.labels = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
        
    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, x):
        outputs = self.model(x)
        return outputs

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        losses, sum_loss, preds = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("train/loss", sum_loss)
        
        for i, name in enumerate(self.labels):
            self.log(f"train/{name}_loss", losses[i])
        
        return {"loss": sum_loss, "preds": preds}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        losses, sum_loss, preds = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log("val/loss", sum_loss)
        
        for i, name in enumerate(self.labels):
            self.log(f"val/{name}_loss", losses[i])

        # Let's return preds to use it in a custom callback
        return {"preds": preds}

    def _get_preds_loss_accuracy(self, batch):
        """convenience function since train/valid/test steps are similar"""
        x = batch
        preds = self(x['input_ids'])

        losses = []
        sum_loss = 0
        for i, name in enumerate(self.labels):
            temp_loss = self.loss(preds[i], x[name].unsqueeze(1))
            losses.append(temp_loss)        
            sum_loss += temp_loss
            
            preds[i] = preds[i].detach()

        return losses, sum_loss, preds