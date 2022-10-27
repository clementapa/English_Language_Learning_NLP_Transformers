import os.path as osp

import torch.nn as nn
from transformers import AutoModel

from model.pooling import CLSPooling, MaxPooling, MeanMaxPooling, MeanPooling


class Model(nn.Module):
    def __init__(
        self,
        name_model,
        nb_of_linears,
        layer_norm,
        pooling,
        last_layer_reinitialization,
        gradient_checkpointing,
        save_pretrained,
    ):
        super(Model, self).__init__()

        # Dropout is bad for regression tasks
        # https://www.kaggle.com/competitions/commonlitreadabilityprize/discussion/260729
        deactivate_dropout = {
            "attention_probs_dropout_prob": 0,
            "hidden_dropout_prob": 0,
            "pooler_dropout": 0,
        }

        if osp.isdir(save_pretrained):
            self.features_extractor = AutoModel.from_pretrained(
                save_pretrained, **deactivate_dropout
            )
        else:
            self.features_extractor = AutoModel.from_pretrained(
                name_model, **deactivate_dropout
            )
            self.features_extractor.save_pretrained(save_pretrained)

        if last_layer_reinitialization:
            for encoder_block in self.features_extractor.base_model.encoder.layer[-1:]:
                for layer in encoder_block.modules():
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight.data)
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias.data, 0)
                    elif isinstance(layer, nn.LayerNorm):
                        nn.init.constant_(layer.weight.data, 1)
                        nn.init.constant_(layer.bias.data, 0)

        if gradient_checkpointing:
            self.features_extractor.gradient_checkpointing_enable()

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
