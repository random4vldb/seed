import collections
import json
import random

import inflect
import jsonlines
import pandas as pd
import pyrootutils
from loguru import logger
from tango import JsonFormat, Step
from transformers import AutoTokenizer
from .helpers import *
from pathlib import Path
from typing import Optional
from tango.common.dataset_dict import DatasetDict
import torch

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)
inflect = inflect.engine()


@Step.register("infotab::json_to_para")
class InfoTabJsonToPara(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT = JsonFormat()
    VERSION: Optional[str] = "005452"

    def run(self, dataset_dict, rand_perm):
        for split, data in dataset_dict.items():
            result = []

            if rand_perm == 2:
                table_ids = []
                for index, row in enumerate(data):
                    table_ids += [row["table_id"]]
                random.shuffle(table_ids)
                for index, row in enumerate(data):
                    row["table_id"] = table_ids[index]

            if rand_perm == 1:
                table_ids = []
                for index, row in enumerate(data):
                    table_ids += [row["table_id"]]

                set_of_orignal = list(set(table_ids))
                set_of_random = set_of_orignal
                random.shuffle(set_of_random)
                set_of_orignal = list(set(table_ids))
                random_mapping_tableids = {}
                jindex = 0

                for key in set_of_orignal:
                    random_mapping_tableids[key] = set_of_random[jindex]
                    jindex += 1

                for index, row in enumerate(data):
                    table_id = row["table_id"]
                    row["table_id"] = random_mapping_tableids[table_id]

            for index, row in enumerate(data):
                if isinstance(row["table"], str):
                    row["table"] = json.loads(row["table"])
                table = pd.DataFrame(row["table"]).astype(str)
                obj = collections.defaultdict(list)
                if row["highlighted_cells"]:
                    for i, j in row["highlighted_cells"]:
                        obj[f"{table.columns[j]}"].append(table.iloc[i, j])
                else:
                    for i in range(table.shape[0]):
                        for j in range(table.shape[1]):
                            obj[f"{table.columns[j]}"].append(table.iloc[i, j])

                if not obj:
                    continue
                obj = {x: y if isinstance(y, list) else [y] for x, y in obj.items()}
                
                try:
                    title = row["title"]
                except KeyError as e:
                    logger.error(f"KeyError: {e}")
                    exit()

                para = ""

                if "index" in obj:
                    obj.pop("index")

                for key in obj:
                    line = ""
                    values = obj[key]

                    if not key.strip():
                        key = "value"
                    key = key.replace(" of ", " ")
                    key = key.replace(
                        " A ", " AB "
                    )  # This is a hack to for inflect to work properly
                    try:
                        res = inflect.plural_noun(key)
                    except:
                        res = None
                    if (len(values) > 1) and res is not None:
                        verb_use = "are"
                        if is_date("".join(values)):
                            para += title + " was " + str(key) + " on "
                            line += title + " was " + str(key) + " on "
                        else:
                            try:
                                para += (
                                    "The "
                                    + str(key)
                                    + " of "
                                    + title
                                    + " "
                                    + verb_use
                                    + " "
                                )
                                line += (
                                    "The "
                                    + str(key)
                                    + " of "
                                    + title
                                    + " "
                                    + verb_use
                                    + " "
                                )
                            except TypeError as e:
                                logger.error(
                                    "Error in key: %s in article of title %s",
                                    key,
                                    title,
                                )
                                exit()
                        for value in values[:-1]:
                            para += value + ", "
                            line += value + ", "
                        if len(values) > 1:
                            para += "and " + values[-1] + ". "
                            line += "and " + values[-1] + ". "
                        else:
                            para += values[-1] + ". "
                            line += values[-1] + ". "
                    else:
                        verb_use = "is"
                        if is_date(values[0]):
                            para += (
                                title + " was " + str(key) + " on " + values[0] + ". "
                            )
                            line += (
                                title + " was " + str(key) + " on " + values[0] + ". "
                            )
                        else:
                            para += (
                                "The "
                                + str(key)
                                + " of "
                                + title
                                + " "
                                + verb_use
                                + " "
                                + values[0]
                                + ". "
                            )
                            line += (
                                "The "
                                + str(key)
                                + " of "
                                + title
                                + " "
                                + verb_use
                                + " "
                                + values[0]
                                + ". "
                            )

                label = row["label"]
                obj = {
                    "index": index,
                    "table_id": row["table_id"],
                    "annotator_id": row["annotator_id"],
                    "premise": para,
                    "hypothesis": row["hypothesis"],
                    "label": label,
                }
                result.append(obj)
            dataset_dict[split] = result
        print(dataset_dict.keys())
        return dataset_dict


@Step.register("infotab::preprocess")
class InfoTabPreprocess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    VERSION: Optional[str] = "0051"

    def run(self, dataset_dict, tokenizer, single_sentence):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Process for every split
        # Initialize dictionary to store processed information
        keys = ["uid", "encodings", "attention_mask", "segments", "labels"]
        for split, data in dataset_dict.items():
            examples = []
            samples_processed = 0
            # Iterate over all data points
            for pt_dict in data:

                samples_processed += 1
                # Encode data. The premise and hypothesis are encoded as two different segments. The
                # maximum length is chosen as 504, i.e, 500 sub-word tokens and 4 special characters
                # If there are more than 504 sub-word tokens, sub-word tokens will be dropped from
                # the end of the longest sequence in the two (most likely the premise)
                if single_sentence:
                    encoded_inps = tokenizer(
                        pt_dict["hypothesis"],
                        padding="max_length",
                        truncation=True,
                        max_length=504,
                        return_tensors="pt",
                    )
                else:
                    encoded_inps = tokenizer(
                        pt_dict["premise"],
                        pt_dict["hypothesis"],
                        padding="max_length",
                        truncation=True,
                        max_length=504,
                        return_tensors="pt",
                    )

                # Some models do not return token_type_ids and hence
                # we just return a list of zeros for them. This is just
                # required for completeness.
                if "token_type_ids" not in encoded_inps.keys():
                    encoded_inps["token_type_ids"] = torch.zeros_like(
                        encoded_inps["input_ids"]
                    )

                examples.append(
                    {
                        "input_ids": encoded_inps["input_ids"].squeeze(0),
                        "attention_mask": encoded_inps["attention_mask"].squeeze(0),
                        "token_type_ids": encoded_inps["token_type_ids"].squeeze(0),
                        "labels": pt_dict["label"],
                    }
                )

            print("Preprocessing Finished")
            dataset_dict[split] = examples
        return DatasetDict(dataset_dict)


@Step.register("infotab::input_data")
class InfoTabInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    VERSION: Optional[str] = "0051"

    def run(self, input_dir):
        split2examples = []
        for input_file in Path(input_dir).glob("*.json"):
            with jsonlines.open(input_file, "r") as reader:
                idx = 0
                for jobj in reader:
                    example = {
                        "table_id": idx,
                        "annotator_id": idx,
                        "hypothesis": jobj["sentence"],
                        "table": jobj["table"],
                        "label": jobj["label"],
                        "title": jobj["table_page_title"]
                        if "table_page_title" in jobj
                        else jobj["title"],
                        "highlighted_cells": jobj["highlighted_cells"]
                    }
                    idx += 1
                    split2examples[input_file.stem].append(example)
            logger.info("Num examples", len(split2examples[input_file.stem]))
        return split2examples


@Step.register("infotab::input_from_totto")
class InfoTabInputFromTotto(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = False
    VERSION = "00791"

    def run(self, input_dir, task):
        idx = 0
        if task == "verification":
            split2examples = collections.defaultdict(list)
            print(input_dir)
            for input_file in Path(input_dir).glob("*.jsonl"):
                print(input_file)
                with jsonlines.open(input_file, "r") as reader:
                    for obj in list(reader):
                        split2examples[input_file.stem].append(
                            {
                                "table_id": idx,
                                "annotator_id": idx,
                                "hypothesis": obj["sentence"],
                                "label": 1,
                                "table": obj["positive_table"],
                                "title": obj["title"],
                                "highlighted_cells": obj["highlighted_cells"],
                            }
                        )
                        negative_table = pd.DataFrame(json.loads(obj["negative_table"]))
                        replacing_value, replaced_value, row, column = json.loads(
                            obj["note"]
                        )
                        negative_table.iloc[row, column] = replacing_value
                        split2examples[input_file.stem].append(
                            {
                                "table_id": idx + 1,
                                "annotator_id": idx + 1,
                                "hypothesis": obj["sentence"],
                                "label": 0,
                                "table": negative_table.to_json(orient="records"),
                                "title": obj["title"],
                                "highlighted_cells": obj["highlighted_cells"],
                            }
                        )
                    idx += 2
            print({k: len(v) for k, v in split2examples.items()})
            return split2examples
        else:
            split2examples = collections.defaultdict(list)
            print(input_dir)
            for input_file in Path(input_dir).glob("*.jsonl"):
                print(input_file)
                with jsonlines.open(input_file, "r") as reader:
                    idx = 0
                    for jobj in reader:
                        print(jobj.keys())

                        example = {
                            "table_id": idx,
                            "annotator_id": idx,
                            "hypothesis": jobj["positive"],
                            "table": jobj["table"],
                            "label": 1,
                            "title": jobj["table_page_title"]
                            if "table_page_title" in jobj
                            else jobj["title"],
                            "highlighted_cells": jobj["highlighted_cells"]
                        }
                        idx += 1
                        split2examples[input_file.stem].append(example)

                        example = {
                            "table_id": idx,
                            "annotator_id": idx,
                            "hypothesis": jobj["negative"],
                            "table": jobj["table"],
                            "label": 0,
                            "title": jobj["table_page_title"]
                            if "table_page_title" in jobj
                            else jobj["title"],
                            "highlighted_cells": jobj["highlighted_cells"]
                        }
                        idx += 1

                        split2examples[input_file.stem].append(example)

                logger.info("Num examples", len(split2examples[input_file.stem]))
            print({k: len(v) for k, v in split2examples.items()})
            return split2examples
