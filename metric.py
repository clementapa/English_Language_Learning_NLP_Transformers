from torchmetrics import Metric
import torch


class MCRMSE(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state(
            "mcrmse",
            default=torch.zeros(1, device=torch.device("cpu")),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "rmse_per_classes",
            default=torch.zeros(6, device=torch.device("cpu")),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "total",
            default=torch.zeros(1, device=torch.device("cpu")),
            dist_reduce_fx="sum",
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        colwise_rmse = torch.sqrt(torch.mean(torch.square(target - preds), dim=0))
        mcrmse = torch.mean(colwise_rmse, dim=0)

        self.mcrmse += mcrmse
        self.rmse_per_classes += colwise_rmse
        self.total += 1

    def compute(self):
        return {
            "avg": self.mcrmse / self.total,
            "per_cls": self.rmse_per_classes / self.total,
        }
