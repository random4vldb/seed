import collections
import json
import random

import inflect
import jsonlines
import pandas as pd
import pyrootutils
from loguru import logger
from tango import JsonFormat, Step, Format
from transformers import AutoTokenizer
from .helpers import *
from pathlib import Path
from typing import Optional
from tango.common.dataset_dict import DatasetDict

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
    VERSION: Optional[str] = "007"

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
                obj = infotab_linearize(index, row)
                if obj is None:
                    continue
                result.append(obj)
            dataset_dict[split] = result
        return dataset_dict


@Step.register("infotab::preprocess")
class InfoTabPreprocess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    VERSION: Optional[str] = "007"

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
                encoded_inps = infotab_tokenize(tokenizer, pt_dict, single_sentence)

                examples.append(
                    {
                        "input_ids": encoded_inps["input_ids"].squeeze(0),
                        "attention_mask": encoded_inps["attention_mask"].squeeze(0),
                        "token_type_ids": encoded_inps["token_type_ids"].squeeze(0),
                        "labels": pt_dict["label"],
                    }
                )

            dataset_dict[split] = examples
        return DatasetDict(dataset_dict)


@Step.register("infotab::input_data")
class InfoTabInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    VERSION: Optional[str] = "007"

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
    VERSION = "009"

    def run(self, input_dir, task):
        idx = 0
        if task == "verification":
            split2examples = collections.defaultdict(list)
            for input_file in Path(input_dir).glob("*.jsonl"):
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
            return split2examples
        else:
            split2examples = collections.defaultdict(list)
            for input_file in Path(input_dir).glob("*.jsonl"):
                with jsonlines.open(input_file, "r") as reader:
                    idx = 0
                    for jobj in list(reader):
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
            return split2examples


@Step.register("infotab::pipeline_preprocess")
class InfotabPipelinePreProcess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, data, doc_results):
        examples = []
        for idx, (example, doc_result) in enumerate(zip(data, doc_results)):
            for sentence in doc_result:
                examples.append(
                    {
                        "table_id": idx,
                        "annotator_id": idx,
                        "hypothesis":  sentence,
                        "table": example["table"],
                        "title": example["title"],
                        "highlighted_cells": example["highlighted_cells"],
                    }
                )

        return examples

@Step.register("infotab::pipeline_postprocess")
class InfotabPipelinePostProcess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, results):
        id2results = {}
        for idx, result in enumerate(results):
            if result["table_id"] in id2results:
                if results["label"] == 1:
                    id2results[result["table_id"]] = True
                else:
                    id2results[result["table_id"]] = False
            else:
                if results["label"] == 1:
                    id2results[result["table_id"]] = True
                
        final_results = [False] * len(results)
        for idx, result in id2results.items():
            final_results[idx] = id2results[idx]
        
        return final_results



@Step.register("infotab::document_retrieval")
class InfotabSentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, data, sentence_results):
        for example, sentence_result in zip(data, sentence_results):
            for doc, score, title in sentence_result:
                if title == example["title"]:
                    example["doc"] = doc
                    example["score"] = score
                    break
    