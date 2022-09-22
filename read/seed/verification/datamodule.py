import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


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

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

        self.train_dev_dataset = datasets.load_dataset(
            "json",
            data_files={
                "train": f"{self.dataset_name_or_path}/train.jsonl",
                "dev": f"{self.dataset_name_or_path}/dev.jsonl",
            },
        )
        self.test_dataset = datasets.load_dataset(
            "json", data_files={"test": self.test_dataset_name_or_path}
        )["test"]

        remove_columns = self.train_dev_dataset["train"].column_names
        self.train_dev_dataset = self.train_dev_dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=remove_columns,
            num_proc=20,
        )

        test_remove_columns = self.test_dataset.column_names

        self.test_dataset = self.test_dataset.map(
            self.convert_to_test_features,
            batched=True,
            remove_columns=test_remove_columns,
            num_proc=20,
        )

        for split in self.train_dev_dataset.keys():
            self.train_dev_dataset[split].set_format(type="torch")

        self.test_dataset.set_format(type="torch")

    def train_dataloader(self):
        return DataLoader(
            self.train_dev_dataset["train"],
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.train_dev_dataset["train"],
            batch_size=self.eval_batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=4,
        )

    def convert_to_test_features(self, example_batch):
        texts_or_text_pairs = (
            list(
                zip(
                    example_batch["positive"],
                    example_batch["sentence"],
                )
            ),
            list(
                zip(
                    example_batch["negative"],
                    example_batch["sentence"],
                )
            ),
        )

        # Tokenize the text/text pairs
        assert len(texts_or_text_pairs[0]) == len(texts_or_text_pairs[1])
        features = self.tokenizer(
            texts_or_text_pairs[0] + texts_or_text_pairs[1],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation="only_first",
        )

        features["labels"] = [1] * len(texts_or_text_pairs[0]) + [0] * len(
            texts_or_text_pairs[1]
        )

        return features

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        texts_or_text_pairs = (
            list(
                zip(
                    example_batch["positive"],
                    example_batch["sentence"],
                )
            ),
            list(
                zip(
                    example_batch["negative"],
                    example_batch["sentence"],
                )
            ),
        )

        # Tokenize the text/text pairs
        assert len(texts_or_text_pairs[0]) == len(texts_or_text_pairs[1])
        positives = self.tokenizer(
            texts_or_text_pairs[0] + texts_or_text_pairs[1],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation="only_first",
        )
        result = {}
        for key in positives.data.keys():
            result[f"positive_{key}"] = positives[key][: len(texts_or_text_pairs[0])]
            result[f"negative_{key}"] = positives[key][len(texts_or_text_pairs[0]) :]
        return result
