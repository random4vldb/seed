import datasets
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tango.integrations.pytorch_lightning import LightningDataModule


@LightningDataModule.register("seed::verification_data")
class VerificationDataModule(LightningDataModule):
    def __init__(
        self,
        tokenizer: str,
        dataset_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.dataset_name_or_path = dataset_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

        self.dataset = datasets.load_dataset(
            "json",
            data_files={
                "train": f"{self.dataset_name_or_path}/train.jsonl",
                "dev": f"{self.dataset_name_or_path}/dev.jsonl",
            },
        )

        remove_columns = self.dataset["train"].column_names
        self.dataset = self.dataset.map(
            self.convert_to_features,
            batched=True,
            remove_columns=remove_columns,
        )

        for split in self.dataset.keys():
            self.dataset[split].set_format(type="torch")

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
            batch_size=self.eval_batch_size,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["dev"],
            batch_size=self.eval_batch_size,
            num_workers=4,
        )


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
        inputs = self.tokenizer(
            texts_or_text_pairs[0] + texts_or_text_pairs[1],
            max_length=self.max_seq_length,
            padding="max_length",
            truncation="only_first",
        )
        inputs["labels"] = [1] * len(texts_or_text_pairs[0]) + [0] * len(texts_or_text_pairs[1])
        return inputs
