from torchmetrics import Metric
import torch


class MCRMSE(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state("mcrmse", default=torch.tensor(0, device=torch.device("cpu")), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0, device=torch.device("cpu")), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        colwise_mse = torch.mean(torch.square(target - preds), dim=0)
        loss = torch.mean(torch.sqrt(colwise_mse), dim=0)

        self.mcrmse += loss.long()
        self.total += target.shape[0]

    def compute(self):
        return self.mcrmse.float() / self.total
