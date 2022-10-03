from transformers import AutoModel
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, name_model):
        super(Model, self).__init__()
        self.features_extractor = AutoModel.from_pretrained(name_model)
        num_features = len(self.features_extractor(self.features_extractor.dummy_inputs['input_ids'])['pooler_output'][0])
        self.cls_list = nn.ModuleList([nn.Linear(num_features, 1) for i in range(6)])

    def forward(self, x):
        features = self.features_extractor(x)['pooler_output']
        outputs = [cls(features) for cls in self.cls_list]
        return outputs