import pytorch_lightning as pl

from metric import MCRMSE


class MetricCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.rmse_train = MCRMSE().cpu()
        self.rmse_val = MCRMSE().cpu()

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
        self.rmse_val(outputs["preds"].cpu(), targets["labels"].cpu())

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        rmse_val = self.rmse_val.compute()
        self.rmse_val.reset()
        self.log("val/mcrmse", rmse_val['avg'])

        for i in range(len(pl_module.labels)):
            self.log(f"val/rmse_{pl_module.labels[i]}", rmse_val['per_cls'][i])

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
        self.rmse_train(outputs["preds"].cpu(), targets["labels"].cpu())

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        rmse_train = self.rmse_train.compute()
        self.rmse_train.reset()
        self.log("train/mcrmse", rmse_train['avg'])
        for i in range(len(pl_module.labels)):
            self.log(f"train/rmse_{pl_module.labels[i]}", rmse_train['per_cls'][i])
