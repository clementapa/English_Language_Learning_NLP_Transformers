import torch
from operator import itemgetter


class EssayDataset:
    def __init__(self, df, max_length, tokenizer=None, is_test=False):
        self.df = df
        self.classes = [
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ]
        self.max_len = max_length
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __getitem__(self, idx):
        sample = self.df[idx]["full_text"]
        tokenized = self.tokenizer.encode_plus(
            sample,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        )
        inputs = {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "token_type_ids": torch.tensor(
                tokenized["token_type_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
        }

        if self.is_test == True:
            return inputs

        label = itemgetter(*self.classes)(self.df[idx])
        targets = {
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        return inputs, targets

    def __len__(self):
        return len(self.df)
