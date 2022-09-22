from pathlib import Path
from typing import Optional

import datasets
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import RagTokenizer, RagTokenForGeneration, DPRQuestionEncoder
from read.dpr.searcher import DPRSearcher
import torch


class DPRDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_name_or_path: str,
        rag_model_name: str,
        rag_tokenizer: str,
        qry_encoder_path: str,
        ctx_encoder_path: str,
        batch_size: int,
        eval_splits: Optional[list] = None,
        test_splits: Optional[list] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name_or_path = dataset_name_or_path
        self.tokenizer = RagTokenizer.from_pretrained(rag_tokenizer, use_fast=True)

        self.model = RagTokenForGeneration.from_pretrained(rag_model_name)

        qencoder = DPRQuestionEncoder.from_pretrained(qry_encoder_path)

        rag_qenocder = self.model.question_encoder
        rag_qenocder.load_state_dict(qencoder.state_dict(), strict=True)

        # self.rag_config: RagConfig = self.model.config

        self.searcher = DPRSearcher()

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset(
            "json",
            data_files={
                "train": f"{self.dataset_name_or_path}/train.json",
                "dev": f"{self.dataset_name_or_path}/dev.json",
            },
        )
        columns = self.dataset.column_names

        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            batch_size=self.hparams.batch_size,
            remove_columns=columns,
            num_proc=20,
        )
        for split in self.dataset:
            self.dataset[split].set_format(type="torch")

    def prepare_data(self):
        dataset = datasets.load_dataset(
            "json",
            data_files={
                "train": f"{self.dataset_name_or_path}/train.json",
                "dev": f"{self.dataset_name_or_path}/dev.json",
            },
        )
        Path(f".cache/huggingface/{self.dataset_name_or_path}").mkdir(
            parents=True, exist_ok=True
        )
        print("Data mapping for", self.trainer.local_rank)
        columns = dataset.column_names
        dataset.map(
            self.convert_to_features,
            batched=True,
            batch_size=self.hparams.batch_size,
            remove_columns=columns,
            num_proc=20,
        )

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

    def convert_to_features(self, batch, indices=None):
        features = self.tokenizer(
            batch["query"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=self.hparams.max_content_length,
        )

        question_enc_outputs = self.model.rag.question_encoder(
            features["input_ids"],
            attention_mask=features["attention_mask"],
            return_dict=True,
        )
        question_encoder_last_hidden_state = question_enc_outputs[0]
        outputs, scores, vectors = self.searcher.batch_query(
            batch["query"], include_vectors=True
        )

        doc_vectors = np.frombuffer([], dtype=self.rest_dtype).reshape(
            -1, self.hparams.k, question_encoder_last_hidden_state.shape[-1]
        )[:, 0 : self.hparams.k, :]
        retrieved_doc_embeds = torch.Tensor(doc_vectors.copy())
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1),
            retrieved_doc_embeds.transpose(1, 2),
        ).squeeze(1)

        context_inputs = self.postprocess_doc(outputs, batch["query"], scores, vectors)
        answers = ["True" if label == 1 else "False" for label in batch["labels"]]

        with self.tokenizer.as_target_tokenizer():
            context_inputs["labels"] = self.tokenizer(
                answers,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=self.hparams.max_content_length,
            )["input_ids"]
        
        return context_inputs, doc_scores, outputs

    def postprocess_docs(
        self, docs, input_strings, prefix=None, return_tensors=None
    ):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            # TODO(Patrick): if we train more RAG models, I want to put the input first to take advantage of effortless truncation
            # TODO(piktus): better handling of truncation
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (
                prefix
                + doc_title
                + self.config.title_sep
                + doc_text
                + self.config.doc_sep
                + input_string
            ).replace("  ", " ")
            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(self.hparams.k)
        ]

        contextualized_inputs = self.tokenizer.generator.batch_encode_plus(
            rag_input_strings,
            max_length=self.model.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs
