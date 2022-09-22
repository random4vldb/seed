from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import multiprocessing
import pandas as pd
import json
from pathlib import Path
from json.decoder import JSONDecodeError


class Seed3DataModule(LightningDataModule):

    def __init__(
        self,
        tokenizer: str,
        dataset_name_or_path: str,
        test_dataset_name_or_path: str,
        max_seq_length: int = 256,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name_or_path = dataset_name_or_path
        self.test_dataset_name_or_path = test_dataset_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer, use_fast=True
        )

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("json", self.dataset_name_or_path))
        
        remove_columns = self.dataset["train"].column_names
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=remove_columns,
            num_proc=20,
            cache_file_names={
                x: f".cache/huggingface/{self.dataset_name_or_path}/cache_{x}.arrow"
                for x in self.dataset
            },
        )

        for split in self.dataset.keys():
            self.dataset[split].set_format(type="torch")


    def prepare_data(self):
        dataset = datasets.load_dataset("json", self.dataset_name_or_path))
        
        remove_columns = dataset["train"].column_names
        dataset = dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=remove_columns,
            num_proc=20,
            cache_file_names={
                x: f".cache/huggingface/{self.dataset_name_or_path}/cache_{x}.arrow"
                for x in dataset
            },
        )


    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["dev"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        ) 


    def convert_to_test_features(self, example_batch):
        tables = []
        for table in example_batch["table"]:
            tables.append(linearize)

        inputs = self.tokenizer(
            example_batch["table"],
        )
