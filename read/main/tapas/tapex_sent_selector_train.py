import json

import hydra
import pandas as pd
import pyrootutils
import torchmetrics
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    TapexTokenizer,
    Trainer,
    TrainingArguments,
)
from loguru import logger
import evaluate 
import numpy as np

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


def sub_table(table, highlighted_cells):
    table = pd.DataFrame(json.loads(table))
    cells = zip(*highlighted_cells)
    cells = [list(x) for x in cells]
    sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)
    return sub_table


def tokenize(batch, tokenizer):
    tables = [
        sub_table(item[0], item[1])
        for item in zip(batch["table"], batch["highlighted_cells"])
    ]

    positive_encodings = tokenizer(
        tables,
        batch["positive"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    positive_encodings["labels"] = [1] * len(positive_encodings["input_ids"])

    negative_encodings = tokenizer(
        tables,
        batch["negative"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    negative_encodings["labels"] = [0] * len(positive_encodings["input_ids"])

    encodings = {}

    for key in positive_encodings:
        encodings[key] = positive_encodings[key] + negative_encodings[key]

    return encodings


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "tapas",
    config_name="tapex_sent_train.yaml",
)
def main(cfg):
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/tapex-base")

    datasets = load_dataset(
        "json", data_files={"train": cfg.train_file, "dev": cfg.dev_file}
    )

    datasets = datasets.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=[
            "query",
            "positive",
            "negative",
            "title",
            "table",
            "highlighted_cells",
            "id",
        ],
    )

    name2metric = {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
    }

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions[0])
        print(predictions, labels)
        for name, metric in name2metric.items():
            metric.add_batch(predictions=predictions, references=labels)
        return {name: metric.compute() for name, metric in name2metric.items()}

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy="epoch",
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_epochs,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["dev"],
        compute_metrics=compute_metrics,
    )
    if cfg.get("train"):
        trainer.train()


        for name, metric in name2metric.items():
            logger.info(f"{name}: {metric.compute()}")
            metric.reset()
    if cfg.get("dev"):

        trainer.evaluate()

        model = model.from_pretrained(cfg.output_dir)

        for name, metric in name2metric.items():
            logger.info(f"{name}: {metric.compute()}")
            metric.reset()


if __name__ == "__main__":
    main()
