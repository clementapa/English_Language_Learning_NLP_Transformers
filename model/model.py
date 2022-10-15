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
    def __init__(self, name_model, nb_of_linears):
        super(Model, self).__init__()
        self.features_extractor = AutoModel.from_pretrained(name_model)
        num_features = self.features_extractor(
            self.features_extractor.dummy_inputs["input_ids"]
        )["last_hidden_state"].shape[-1]

        self.pooler = MeanPooling()

        if nb_of_linears != 0:
            temp_num_features = num_features
            self.linears = nn.ModuleList()
            for i in range(nb_of_linears):
                self.linears.append(nn.Linear(temp_num_features, temp_num_features // 2))
                self._initialize_weights(self.linears[i])
                temp_num_features = temp_num_features // 2
            num_features = temp_num_features

        self.cls = nn.Linear(num_features, 6)
        self._initialize_weights(self.cls)

    def forward(self, inputs):
        outputs = self.features_extractor(**inputs, return_dict=True)
        features = self.pooler(outputs["last_hidden_state"], inputs["attention_mask"])
        if hasattr(self, "linears"):
            for layer in self.linears:
                features = layer(features)
        outputs = self.cls(features)
        return outputs
    
    def _initialize_weights(self, m):
        nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)        

