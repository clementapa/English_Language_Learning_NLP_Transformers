import os, errno
import torch
from torch.nn.utils.rnn import pad_sequence


def create_dir(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def collate_batch(batch):

    inputs = {
        "input_ids": [],
        "token_type_ids": [],
        "attention_mask": [],
    }

    with_labels = len(batch[0]) == 2

    if with_labels:
        targets = {"labels": []}

    for b in batch:
        inputs["input_ids"].append(b[0]["input_ids"])
        inputs["token_type_ids"].append(b[0]["token_type_ids"])
        inputs["attention_mask"].append(b[0]["attention_mask"])
        if with_labels:
            targets["labels"].append(b[1]["labels"])

    inputs["input_ids"] = pad_sequence(
        inputs["input_ids"], batch_first=True, padding_value=0
    )
    inputs["token_type_ids"] = pad_sequence(
        inputs["token_type_ids"], batch_first=True, padding_value=0
    )
    inputs["attention_mask"] = pad_sequence(
        inputs["attention_mask"], batch_first=True, padding_value=0
    )

    if with_labels:
        targets["labels"] = torch.stack(targets["labels"])
        return inputs, targets

    return inputs
