import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
import os.path as osp
from sklearn.model_selection import train_test_split
from utils import create_dir

class ELL_data(pl.LightningDataModule):
    def __init__(self, config):
        super(ELL_data, self).__init__()

        self.root = config.root
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers

        self.validation_split = config.validation_split
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["full_text"], padding="max_length", truncation=True)

    def format_set(self, dataset):
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True)
        tokenized_dataset.set_format(type='torch', columns=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions', 'input_ids', 'token_type_ids', 'attention_mask'])
        return tokenized_dataset

    def prepare_data(self) -> None:
        data_dir = osp.join('assets', f'dataset_train_val_{self.validation_split}.hf')
        if not osp.isdir(data_dir):
            dataset = load_dataset("csv", data_files=osp.join(self.root, "feedback-prize-english-language-learning/train.csv"))
            train, val = train_test_split(dataset['train'], test_size=self.validation_split, random_state=13)

            create_dir(data_dir)
            dataset['train'] = self.format_set(Dataset.from_dict(train))
            dataset['val'] = self.format_set(Dataset.from_dict(val))
            self.dataset = dataset
            self.dataset.save_to_disk(data_dir)
        else:
            self.dataset = load_from_disk(data_dir)
            self.dataset.set_format(type='torch', columns=['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions', 'input_ids', 'token_type_ids', 'attention_mask'])

    def setup(self, stage=None):
        # split dataset
        if stage in (None, "fit"):   
            self.train = self.dataset['train']
            self.val = self.dataset['val']

    def train_dataloader(self):
        train = DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train

    def val_dataloader(self):
        val = DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return val

    def test_dataloader(self):
        test = DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return test

    def predict_dataloader(self):
        predict = DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        return predict