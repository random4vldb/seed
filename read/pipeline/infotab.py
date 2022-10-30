import collections
import json
import random
from typing import Optional
import json
import jsonlines
import random

import jsonlines
import torch
from loguru import logger
from read.baselines.infotab.helpers import infotab_linearize, infotab_tokenize
from read.utils.table import infotab2totto
from tango import Format, JsonFormat, Step
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

@Step.register("pipeline::input_infotab")
class InfotabInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0013"

    def run(self, input_file, size=-1):
        with jsonlines.open(input_file) as reader:
            data = list(reader)
            random.seed(21)
            random.shuffle(data)
            data = data
            if size != -1:
                data = data[:size]

        for i in range(len(data)):
            data[i]["linearized_table"] = infotab2totto(data[i])
            data[i]["table"] = json.loads(data[i]["table"])
            data[i]["highlighted_cells"] = [[0, i] for i in range(len(data[i]["table"][0]))]
            data[i]["label"] = data[i]["label"] == 2
        return data


@Step.register("pipeline::infotab_add_sentence")
class InfotabAddSentence(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0071"


    def run(self, data, doc_results):
        for idx, (example, doc_result) in enumerate(zip(data, doc_results)):
            for i, (doc, score, title) in enumerate(doc_result):
                if title in example["title"] or example["title"] in title:
                    print(title, example["title"], title in example["title"], example["title"] in title)
                    doc_result.append((example["sentence"], 0, example["title"]))
                    break
        return doc_results

@Step.register("infotab::sentence_selection")
class InfotabSentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0027"

    def run(self, model, tokenizer, data, doc_results, batch_size=4):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        infotab_data = []
        sentences = []
        for idx, (example, sentence_result) in enumerate(list(zip(data, doc_results))[:100]):
            for sent, score, title in sentence_result:
                infotab_data.append(
                    {
                        "table_id": idx,
                        "annotator_id": idx,
                        "hypothesis":  sent,
                        "table": example["table"],
                        "title": example["title"],
                        "highlighted_cells": example["highlighted_cells"],
                        "label": 0
                    }
                )
                sentences.append(sent)

        all_preds = []
        all_scores = []
        logger.info(f"Running inference on Infotab with {len(infotab_data)} examples")
        model = model.cuda()
        encoded_data = collections.defaultdict(list)
        for idx, example in tqdm(enumerate(infotab_data)):
            linearized = infotab_linearize(idx, example)
            encoded_inps = infotab_tokenize(tokenizer, linearized, single_sentence=True)
            
            for key, value in encoded_inps.items():
                encoded_data[key].append(value.squeeze(0))
            encoded_data["labels"].append(torch.tensor([0]))
        

        dataset = TensorDataset(
            torch.stack(encoded_data["input_ids"]),
            torch.stack(encoded_data["attention_mask"]),
            torch.stack(encoded_data["token_type_ids"]),
            torch.stack(encoded_data["labels"]),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in tqdm(dataloader):
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            outputs = model(**inputs)
            logits = outputs["outputs"]
            all_preds.extend(outputs["predictions"].detach().cpu().numpy().tolist())
            all_scores.extend(torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist())

        assert len(all_preds) == len(all_scores) == len(infotab_data)
        sentence_results = [[[], [], []] for _ in range(len(doc_results))]
        for (pred, sent, score, infotab_example) in zip(all_preds, sentences, all_scores, infotab_data):
            id = infotab_example["table_id"]
            if pred == 1:
                sentence_results[id][0].append((sent, score))
            sentence_results[id][1].append((sent, pred))
            sentence_results[id][2] = (
                data[id]["linearized_table"],
                data[id]["sentence"],
            )
        
        return sentence_results
        
        
@Step.register("infotab::table_verification")
class InfotabVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0023"

    def run(self, model, tokenizer, data, sentence_results, batch_size=4):
        infotab_data = []
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        sentence_results = [x[0] for x in sentence_results]
        sentences = []

        for idx, (example, sentence_result) in enumerate(list(zip(data, sentence_results))[:100]):
            for sent, score in sentence_result:
                infotab_data.append(
                    {
                        "table_id": idx,
                        "annotator_id": idx,
                        "hypothesis":  sent,
                        "table": example["table"],
                        "title": example["title"],
                        "highlighted_cells": example["highlighted_cells"],
                        "label": 0
                    }
                )
                sentences.append(sent)

        all_preds = []
        all_scores = []
        logger.info(f"Running inference on Infotab with {len(infotab_data)} examples")
        model = model.cuda()
        encoded_data = collections.defaultdict(list)
        for idx, example in tqdm(enumerate(infotab_data)):
            linearized = infotab_linearize(idx, example)
            encoded_inps = infotab_tokenize(tokenizer, linearized, single_sentence=True)
            
            for key, value in encoded_inps.items():
                encoded_data[key].append(value.squeeze(0))
            encoded_data["labels"].append(torch.tensor([0]))
        

        dataset = TensorDataset(
            torch.stack(encoded_data["input_ids"]),
            torch.stack(encoded_data["attention_mask"]),
            torch.stack(encoded_data["token_type_ids"]),
            torch.stack(encoded_data["labels"]),
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        for batch in tqdm(dataloader):
            batch = tuple(t.cuda() for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2]}
            outputs = model(**inputs)
            logits = outputs["outputs"]
            all_preds.extend(outputs["predictions"].detach().cpu().numpy().tolist())
            all_scores.extend(torch.softmax(logits, dim=1).detach().cpu().numpy().tolist())

        assert len(all_preds) == len(all_scores) == len(infotab_data)
        verified_results = [[False, []] for _ in range(len(sentence_results))]
        for (pred, sent, score, infotab_example) in zip(all_preds,sentences, all_scores, infotab_data):
            id = infotab_example["table_id"]
            if pred == 1:
                verified_results[id][0] = True
            verified_results[id][1].append(
                (
                    sent,
                    score,
                    data[id]["linearized_table"],
                    data[id]["sentence"],
                    data[id]["label"],
                )
            )
        return verified_results 
