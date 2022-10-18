from transformers import AutoModel
import torch.nn as nn
import os.path as osp
from pooling import MeanPooling, MaxPooling, MeanMaxPooling, CLSPooling


class Model(nn.Module):
    def __init__(self, name_model, nb_of_linears, layer_norm, pooling, save_pretrained):
        super(Model, self).__init__()

        if osp.isdir(save_pretrained):
            self.features_extractor = AutoModel.from_pretrained(save_pretrained)
        else:
            self.features_extractor = AutoModel.from_pretrained(name_model)
            self.features_extractor.save_pretrained(save_pretrained)

        num_features = self.features_extractor(
            self.features_extractor.dummy_inputs["input_ids"]
        )["last_hidden_state"].shape[-1]

        if pooling == "MeanPooling":
            self.pooler = MeanPooling()
        elif pooling == "MaxPooling":
            self.pooler = MaxPooling()
        elif pooling == "MeanMaxPooling":
            self.pooler = MeanMaxPooling()
            num_features = num_features * 2
        elif pooling == "CLSPooling":
            self.pooler = CLSPooling()

        if layer_norm:
            self.layer_norm = nn.LayerNorm(num_features)

        if nb_of_linears != 0:
            temp_num_features = num_features
            self.linears = nn.ModuleList()
            for i in range(nb_of_linears):
                self.linears.append(
                    nn.Linear(temp_num_features, temp_num_features // 2)
                )
                self._initialize_weights(self.linears[i])
                temp_num_features = temp_num_features // 2
            num_features = temp_num_features

        self.cls = nn.Linear(num_features, 6)
        self._initialize_weights(self.cls)

    def forward(self, inputs):
        outputs = self.features_extractor(**inputs, return_dict=True)
        features = self.pooler(outputs["last_hidden_state"], inputs["attention_mask"])
        if hasattr(self, "layer_norm"):
            features = self.layer_norm(features)
        if hasattr(self, "linears"):
            for layer in self.linears:
                features = layer(features)
        outputs = self.cls(features)
        return outputs

    def _initialize_weights(self, m):
        nn.init.orthogonal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
