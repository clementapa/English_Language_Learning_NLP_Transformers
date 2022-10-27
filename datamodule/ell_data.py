from pyparsing import FollowedBy
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os.path as osp
from sklearn.model_selection import train_test_split
from datamodule.essay_dataset import EssayDataset
import pandas as pd
from utils import create_dir, collate_batch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


class ELL_data(pl.LightningDataModule):
    def __init__(self, config, fold=None):
        super(ELL_data, self).__init__()

        self.root = config.root
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.config = config

        self.validation_split = config.validation_split

        save_pretrained_tokenizer = osp.join(config.save_pretrained, "tokenizer")
        if osp.isdir(save_pretrained_tokenizer):
            self.tokenizer = AutoTokenizer.from_pretrained(save_pretrained_tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(config.name_model)
            self.tokenizer.save_pretrained(save_pretrained_tokenizer)

        self.fold = fold

        self.labels = [
            "cohesion",
            "syntax",
            "vocabulary",
            "phraseology",
            "grammar",
            "conventions",
        ]

    def prepare_data(self) -> None:
        if not self.config.test:
            if self.config.split_dataset == "normal":
                data_dir = osp.join(
                    "assets", f"dataset_train_val_{self.validation_split}"
                )
                if not osp.isdir(data_dir):
                    dataset = pd.read_csv(
                        osp.join(
                            self.root,
                            "feedback-prize-english-language-learning/train.csv",
                        )
                    )
                    self.train_set, self.val_set = train_test_split(
                        dataset,
                        test_size=self.validation_split,
                        random_state=self.config.random_state,
                    )
                    create_dir(data_dir)
                    self.train_set.to_csv(osp.join(data_dir, "train.csv"), index=False)
                    self.val_set.to_csv(osp.join(data_dir, "val.csv"), index=False)
                else:
                    self.train_set = pd.read_csv(osp.join(data_dir, "train.csv"))
                    self.val_set = pd.read_csv(osp.join(data_dir, "val.csv"))
            elif self.config.split_dataset == "MultilabelStratifiedKFold":
                data_dir = osp.join(
                    "assets", f"dataset_{self.config.N_fold}_MultilabelStratifiedKFold"
                )
                if not osp.isdir(data_dir):
                    dataset = pd.read_csv(
                        osp.join(
                            self.root,
                            "feedback-prize-english-language-learning/train.csv",
                        )
                    )
                    skf = MultilabelStratifiedKFold(
                        n_splits=self.config.N_fold,
                        shuffle=True,
                        random_state=self.config.random_state,
                    )
                    for n, (train_index, val_index) in enumerate(
                        skf.split(dataset, dataset[self.labels])
                    ):
                        dataset.loc[val_index, "fold"] = int(n)
                    dataset["fold"] = dataset["fold"].astype(int)
                    print(dataset["fold"].value_counts())
                    create_dir(data_dir)
                    dataset.to_csv(osp.join(data_dir, "df_folds.csv"), index=False)

                    self.train_set = dataset[dataset["fold"] != self.fold].reset_index(
                        drop=True
                    )
                    self.val_set = dataset[dataset["fold"] == self.fold].reset_index(
                        drop=True
                    )
                else:
                    dataset = pd.read_csv(osp.join(data_dir, "df_folds.csv"))
                    print(dataset["fold"].value_counts())
                    self.train_set = dataset[dataset["fold"] != self.fold].reset_index(
                        drop=True
                    )
                    self.val_set = dataset[dataset["fold"] == self.fold].reset_index(
                        drop=True
                    )
            else:
                raise NotImplementedError(
                    f"split_dataset {self.config.split_dataset} not supported"
                )

        else:
            self.dataset = pd.read_csv(
                osp.join(self.root, "feedback-prize-english-language-learning/test.csv")
            )

    def setup(self, stage=None):
        # split dataset
        if stage in (None, "fit"):
            self.train_set = EssayDataset(
                self.train_set, self.config.max_length, tokenizer=self.tokenizer
            )
            self.val_set = EssayDataset(
                self.val_set, self.config.max_length, tokenizer=self.tokenizer
            )
        else:
            self.predict_set = EssayDataset(
                self.dataset,
                self.config.max_length,
                tokenizer=self.tokenizer,
                is_test=True,
            )

    def train_dataloader(self):
        train = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_batch if self.train_set.max_len == None else None,
        )
        return train

    def val_dataloader(self):
        val = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_batch if self.val_set.max_len == None else None,
        )
        return val

    def predict_dataloader(self):
        predict = DataLoader(
            self.predict_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_batch if self.predict_set.max_len == None else None,
        )
        return predict
