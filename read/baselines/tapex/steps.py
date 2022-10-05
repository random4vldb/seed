from tango import Step
import torch
from datasets import load_dataset
import pandas as pd
import json
from tango.common.dataset_dict import DatasetDict


def sub_table(table, highlighted_cells, note):
    table = pd.DataFrame(json.loads(table))
    if note is not None:
        replacing_value, replaced_value, row, column = json.loads(note)
        table.iloc[row, column] = replacing_value
    cells = zip(*highlighted_cells)
    cells = [list(x) for x in cells]
    sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)
    return sub_table


def tokenize_sent_selection(batch, tokenizer):
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


def tokenize_verification(batch, tokenizer):
    positive_tables = [
        sub_table(item[0], item[1], item[2])
        for item in zip(batch["positive_table"], batch["highlighted_cells"], batch["note"])
    ]

    negative_tables = [
        sub_table(item[0], item[1], item[2])
        for item in zip(batch["negative_table"], batch["highlighted_cells"], batch["note"])
    ]

    positive_encodings = tokenizer(
        positive_tables,
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    positive_encodings["labels"] = [1] * len(positive_encodings["input_ids"])

    negative_encodings = tokenizer(
        negative_tables,
        batch["sentence"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    negative_encodings["labels"] = [0] * len(positive_encodings["input_ids"])

    encodings = {}

    for key in positive_encodings:
        encodings[key] = positive_encodings[key] + negative_encodings[key]

    return encodings


@Step.register("tapex_input_data")
class TapexInputData(Step):
    def run(self, train_file, dev_file, tokenizer, task):
        datasets = load_dataset(
            "json", data_files={"train": train_file, "dev": dev_file}
        )

        tokenize = tokenize_sent_selection if task == "sent_selection" else tokenize_verification

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
        return DatasetDict(
            {
                "train": datasets["train"],
                "dev": datasets["dev"],
            }
        )
