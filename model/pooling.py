'''
    Inspired from this notebook:
    https://www.kaggle.com/code/rhtsingh/utilizing-transformer-representations-efficiently/notebook
'''
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

class MaxPooling(nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        last_hidden_state[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        max_embeddings = torch.max(last_hidden_state, 1)[0]
        return max_embeddings

class MeanMaxPooling(nn.Module):
    def __init__(self):
        super(MeanMaxPooling, self).__init__()
        self.mean_pooling = MeanPooling()
        self.max_pooling = MaxPooling()

    def forward(self, last_hidden_state, attention_mask):
        mean_pooling_embeddings = self.mean_pooling(last_hidden_state, attention_mask)
        max_pooling_embeddings = self.max_pooling(last_hidden_state, attention_mask)
        concat_embeddings = torch.cat((mean_pooling_embeddings, max_pooling_embeddings), 1)
        return concat_embeddings

class CLSPooling(nn.Module):
    def __init__(self):
        super(CLSPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        return last_hidden_state[:, 0]

