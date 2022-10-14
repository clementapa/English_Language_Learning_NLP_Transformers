import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoTokenizer
import os.path as osp
from sklearn.model_selection import train_test_split
from utils import create_dir
from datamodule.essay_dataset import EssayDataset

class ELL_data(pl.LightningDataModule):
    def __init__(self, config):
        super(ELL_data, self).__init__()

        self.root = config.root
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.config = config

        self.validation_split = config.validation_split
        self.tokenizer = AutoTokenizer.from_pretrained(config.name_model)

    def prepare_data(self) -> None:
        data_dir = osp.join('assets', f'dataset_train_val_{self.validation_split}.hf')
        if not osp.isdir(data_dir):
            dataset = load_dataset("csv", data_files=osp.join(self.root, "feedback-prize-english-language-learning/train.csv"))
            train, val = train_test_split(dataset['train'], test_size=self.validation_split, random_state=13)
            dataset['train'] = Dataset.from_dict(train)
            dataset['val'] = Dataset.from_dict(val)
            self.dataset = dataset
            self.dataset.save_to_disk(data_dir)
        else:
            self.dataset = load_from_disk(data_dir)

    def setup(self, stage=None):
        # split dataset
        if stage in (None, "fit"):   
            self.train_set = EssayDataset(self.dataset['train'], self.config.max_length, tokenizer=self.tokenizer)
            self.val_set = EssayDataset(self.dataset['val'], self.config.max_length, tokenizer=self.tokenizer)
        else:
            self.test_set = EssayDataset(self.dataset['test']) 
    def train_dataloader(self):
        train = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train

    def val_dataloader(self):
        val = DataLoader(
            self.val_set,
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