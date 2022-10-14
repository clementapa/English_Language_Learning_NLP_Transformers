from transformers import AutoModel
import torch.nn as nn
import torch


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class Model(nn.Module):
    def __init__(self, name_model):
        super(Model, self).__init__()
        self.features_extractor = AutoModel.from_pretrained(name_model)
        num_features = self.features_extractor(
            self.features_extractor.dummy_inputs["input_ids"]
        )["last_hidden_state"].shape[-1]

        self.pooler = MeanPooling()

        self.linear = nn.Linear(num_features, num_features // 2)
        self.cls_list = nn.ModuleList(
            [nn.Linear(num_features // 2, 1) for i in range(6)]
        )

    def forward(self, inputs):
        outputs = self.features_extractor(**inputs, return_dict=True)
        features = self.pooler(outputs["last_hidden_state"], inputs["attention_mask"])
        features = self.linear(features)
        outputs = [cls(features) for cls in self.cls_list]
        return torch.stack(outputs)
