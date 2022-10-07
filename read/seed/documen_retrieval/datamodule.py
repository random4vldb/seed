import random
from operator import itemgetter
from pathlib import Path
from typing import Optional

import datasets
import jsonlines
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class DPRDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name_or_path: str,
        positive_pid_file: str,
        qry_tokenizer: str,
        ctx_tokenizer: str,
        batch_size: int,
        sample_negative_from_top_k: int,
        seq_len_c: int,
        seq_len_q: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name_or_path = dataset_name_or_path
        self.qry_tokenizer = AutoTokenizer.from_pretrained(qry_tokenizer, use_fast=True)
        self.ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_tokenizer, use_fast=True)
        self.id2pos_pids = {}
        
        with jsonlines.open(positive_pid_file) as reader:
            for jobj in reader:
                self.id2pos_pids[jobj['id']] = jobj['positive_pids']

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("json", data_files=self.dataset_name_or_path, split="train")
        columns = self.dataset.column_names

        self.dataset = self.dataset.filter(lambda x: len(x["negatives"]) > 0).map(
            self.convert_to_features,
            batched=True,
            batch_size=self.hparams.batch_size,
            remove_columns=columns,
            num_proc=20
        ).train_test_split(train_size=0.9)
        for split in self.dataset:
            self.dataset[split].set_format(type="torch")

    def prepare_data(self):
        dataset = datasets.load_dataset("json", data_files=self.dataset_name_or_path, split="train")
        print("Data mapping for", self.trainer.local_rank)
        dataset = dataset.filter(lambda x: len(x["negatives"]) > 0)
        columns = dataset.column_names
        dataset.map(
            self.convert_to_features,
            batched=True,
            batch_size=self.hparams.batch_size,
            remove_columns=columns,
            num_proc=20,
        ).train_test_split(train_size=0.9)
    
    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.hparams.batch_size,
            num_workers=4,
        )

    def create_conflict_free_batch(self, pos_pids, ctx_positive_pids, ctx_negative_pids):
        batch_neg_pids = set()  # the pids that our batch will call batch negatives (for any instance we might add to the batch)
        batch_pos_pids = set()  # the actual positives across all instances in our batch
        nonconflicted_ids = []
        for idx, _ in enumerate(pos_pids):
            # adding it to our batch should not violate our hard negative constraint:
            #  no positive or hard negative for one instance should be a positive for another instance
            if (all([pp not in batch_neg_pids for pp in pos_pids[idx]]) and
                    all([np not in batch_pos_pids for np in ctx_positive_pids[idx]]) and
                    all([np not in batch_pos_pids for np in ctx_negative_pids[idx]])):
                for cp in ctx_positive_pids[idx] + ctx_negative_pids[idx]:
                    batch_neg_pids.add(cp)
                for pp in pos_pids[idx]:
                    batch_pos_pids.add(pp)
                nonconflicted_ids.append(idx)  # this instance can't go in the current batch
        return nonconflicted_ids

    def convert_to_features(self, batch, indices=None):
        qrys = batch["query"]

        positive_titles = [x["title"] for x in batch["positive"]]
        positive_texts = [x["text"] for x in batch["positive"]]
        ctx_positive_pids = [x["pid"] for x in batch["positive"]]

       
        negative_titles = []
        negative_texts = []
        ctx_negative_pids = []
        for x in batch["negatives"]:
            if self.hparams.sample_negative_from_top_k > 0:
                neg_ndx = random.randint(
                    0, min(len(batch["negatives"]), self.hparams.sample_negative_from_top_k) - 1
                )
            else:
                neg_ndx = 0
            negative_texts.append(x[neg_ndx]["text"])
            negative_titles.append(x[neg_ndx]["title"])
            ctx_negative_pids.append(x[neg_ndx]["pid"])
        pos_pids = [self.id2pos_pids[x] for x in batch["id"]]

        nonconflict_indices = self.create_conflict_free_batch(pos_pids, ctx_positive_pids, ctx_negative_pids)
        positive_titles  = list(itemgetter(*nonconflict_indices)(positive_titles))
        positive_texts = list(itemgetter(*nonconflict_indices)(positive_texts))
        ctx_positive_pids = list(itemgetter(*nonconflict_indices)(ctx_positive_pids))
        negative_titles = list(itemgetter(*nonconflict_indices)(negative_titles))
        negative_texts = list(itemgetter(*nonconflict_indices)(negative_texts))
        ctx_negative_pids = list(itemgetter(*nonconflict_indices)(ctx_negative_pids))
        qrys = list(itemgetter(*nonconflict_indices)(qrys))

        positive_inputs = self.ctx_tokenizer(positive_titles + negative_titles, positive_texts + negative_texts, max_length=self.hparams.seq_len_c,
                                          truncation=True, padding="max_length")
        result = {}
        for key in positive_inputs.data.keys():
            if key in ["input_ids", "attention_mask"]:
                result[f"positive_{key}"] = positive_inputs[key][: len(positive_titles)]
                result[f"negative_{key}"] = positive_inputs[key][len(positive_titles):]
        query_inputs = self.qry_tokenizer(qrys, max_length=self.hparams.seq_len_c,
                                          truncation=True, padding="max_length")
        for key in query_inputs.data.keys():
            if key in ["input_ids", "attention_mask"]:
                result[f"query_{key}"] = query_inputs[key]
        return result