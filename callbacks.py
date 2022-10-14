import pytorch_lightning as pl

from metric import MCRMSE


class MetricCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.mcrmse_train = MCRMSE()
        self.mcrmse_val = MCRMSE()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _, targets = batch
        self.mcrmse_val(outputs["preds"], targets["labels"])

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        mcrmse_val = self.mcrmse_val.compute()
        self.mcrmse_val.reset()
        self.log("val/mcrmse", mcrmse_val)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        _, targets = batch
        self.mcrmse_train(outputs["preds"], targets["labels"])

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        mcrmse_train = self.mcrmse_train.compute()
        self.mcrmse_train.reset()
        self.log("train/mcrmse", mcrmse_train)
