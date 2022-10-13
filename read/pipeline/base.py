import collections
import datetime
from typing import Optional

import evaluate
import numpy as np
import pandas as pd
from blingfire import text_to_sentences
from datasets import Dataset
from tango import Format, JsonFormat, Step
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from scipy.special import softmax
import jaro


def tokenize(tokenizer, tokenizer_type, x):
    if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
        table = pd.DataFrame(x["table"]).astype(str)
        inputs = tokenizer(table, x["sent"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    else:
        table = x["table"]
        inputs = tokenizer([(table, x["sent"])], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    return inputs


@Step.register("pipeline::sentence_selection")
class SentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "00597"


    def run(self, model, tokenizer, data, doc_results, batch_size):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

        tables = []
        sentences = []
        ids = []
    
        for idx, (example, doc_result) in enumerate(list(zip(data, doc_results))[:1]):

            for doc, score, title in doc_result:
                for sent in text_to_sentences(doc).split("\n"):
                    if len(sent.split()) <= 4 or sent[-1] != ".": # Short sentences are mostly section titles. 
                            continue
                    if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
                        tables.append(example["table"])
                    else:
                        tables.append(example["linearized_table"])
                    sentences.append(sent)
                    ids.append(idx)
        

        
        training_args = TrainingArguments(
            output_dir="models/sentence_selection",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            label_names=["label"],
            eval_accumulation_steps=8,
        )

        dataset = Dataset.from_dict({"table": tables, "sent": sentences})
        dataset = dataset.map(lambda x: tokenize(tokenizer, tokenizer_type, x), batch_size=32, remove_columns=["table", "sent"])
        dataset.set_format(type="torch")
        

        collator = DataCollatorWithPadding(tokenizer)


        trainer = Trainer(
            args=training_args,
            model=model,
            data_collator=collator,
        )

        outputs = trainer.predict(dataset).predictions

        all_preds = np.argmax(outputs[0], axis=1).tolist()
        scores = softmax(outputs[0], axis=1)[:, 1].tolist()


        assert len(ids) == len(all_preds) == len(scores) == len(sentences)
        sentence_results = [[[],[], []] for _ in range(100)]
        for (pred, sent, score, id) in zip(all_preds, sentences, scores, ids):
            if pred == 1:
                sentence_results[id][0].append((sent, score))
            sentence_results[id][1].append((sent, pred))
            sentence_results[id][2] = (data[id]["linearized_table"], data[id]["sentence"])
        return sentence_results


@Step.register("pipeline::table_verification")
class TableVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "00383"


    def run(self, model, tokenizer, data, sentence_results, batch_size=8):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        sentence_results = [x[0] for x in sentence_results]

        tables = []
        sentences = []
        sent_scores = []
        ids = []
        for i, (example, result) in enumerate(zip(data, sentence_results)):
            for sent, score in result:
                if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
                    tables.append(pd.DataFrame(example["table"]))
                else:
                    tables.append(example["linearized_table"])
                
                sentences.append(sent)
                sent_scores.append(score)
                ids.append(i)
        
        training_args = TrainingArguments(
            output_dir="models/sentence_selection",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            label_names=["label"],
            eval_accumulation_steps=8,
        )

        print(tables)
        dataset = Dataset.from_dict({"table": tables, "sent": sentences})
        dataset = dataset.map(lambda x: tokenize(tokenizer, tokenizer_type, x), batch_size=32, remove_columns=["table", "sent"])
        dataset.set_format(type="torch")
        

        collator = DataCollatorWithPadding(tokenizer)


        trainer = Trainer(
            args=training_args,
            model=model,
            data_collator=collator,
        )

        outputs = trainer.predict(dataset).predictions

        all_preds = np.argmax(outputs[0], axis=1).tolist()
        scores = softmax(outputs[0], axis=1)[:, 1].tolist()

        verified_results = [[False, []] for _ in range(len(sentence_results))]

        for (pred, sent, score, id) in zip(all_preds, sentences, scores, ids):
            print(pred, sent, score, id)
            if pred == 1:
                verified_results[id][0] = True
            verified_results[id][1].append((sent, score, data[id]["linearized_table"], data[id]["sentence"], data[id]["label"]))
        return verified_results


@Step.register("pipeline::cell_correction")
class CellCorrection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "001"


    def choose_question(self, table, column): 
        if "date" in column.lower():
            return "What is the date?"
        if "time" in column.lower():
            return "What time ?"
        try:
            datetime.strptime(table[column][0])
            return "What is the date?"
        except:
            pass

        return "What is the value of " + column + "?"
        

    def run(self, model, tokenizer, data, verified_results, batch_size=8):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)


        tables = []
        sentences = []
        for i, (example, result) in enumerate(zip(data, verified_results)):
            for sent in result:
                if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
                    tables.append(pd.DataFrame(example["table"]))
                else:
                    tables.append(example["linearized_table"])
                
                sentences.append(sent)
        
        dataset = Dataset.from_dict({"table": tables, "sent": sentences})
        dataset = dataset.map(lambda x: tokenize(tokenizer, x), batch_size=1000, num_proc=4, remove_columns=["table", "sent"])
        dataset.set_format(type="torch")

        results = []
        for i, (example, result) in enumerate(zip(data, verified_results)):
            table = pd.DataFrame(example["table"])
            cells = zip(*example["highlighted_cells"])
            cells = [list(x) for x in cells]
            sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)            
            verified_result, sents = result
            if verified_result == 1:
                continue
            for column in sub_table.columns:
                for sent in sents:
                    question = self.choose_question(sub_table, column)
                    input_string = question + "\\n" + sent
                    answer = self.run_model(model, tokenizer, input_string, temperature=0.9, num_return_sequences=1, num_beams=20)
                    if jaro.jaro_winkler_metric(answer, sub_table[column][0]) > 0.8:
                        continue
                    else:
                        results.append(sub_table[column][0])
                        break
                else:
                    results.append(None)
        return results


@Step.register("pipeline::evaluation")
class Evaluation(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "00499"

    step2name2metrics = {
        x : {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }
        for x in ["sentence_selection", "table_verification"]
    }

    step2name2metrics["document_retrieval"] = {
        "retrieval": evaluate.load("trec_eval")
    }

    def jaccard_similarity(self, list1, list2):
        return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))

    def process_sentence_selection(self, data, sentence_results):
        preds = []
        labels = []
        for example, result in zip(data, sentence_results):
            for sent, pred in result:
                if self.jaccard_similarity(sent.split(), example["sentence"].split()) > 0.8:
                    labels.append(1)
                else:
                    labels.append(0)
                preds.append(pred)
        return preds, labels

    def process_document_retrieval(self, data, doc_results):
        preds = []
        labels = []
        for idx, (example, result) in enumerate(zip(data, doc_results)):
            for rank, (doc, score, title) in enumerate(sorted(result, key=lambda x: x[1], reverse=True)):
                preds.append({
                    "query": idx,
                    "q0": "q0",
                    "docid": title,
                    "rank": rank,
                    "score": score,
                    "system": "system"
                })
            labels.append({
                "query": idx,
                "q0": "q0",
                "docid": example["title"],
                "rel": 1
            })

        return [pd.DataFrame(preds).to_dict(orient="list")], [pd.DataFrame(labels).to_dict(orient="list")]

    def run(self, data, doc_results, sentence_results, verified_results):
        sentence_results = [x[1] for x in sentence_results]
        doc_preds, doc_labels = self.process_document_retrieval(data, doc_results)
        sentence_preds, sentence_labels = self.process_sentence_selection(data, sentence_results)

        result = collections.defaultdict(dict)

        for step, preds, labels in zip(["document_retrieval", "sentence_selection", "table_verification"], [doc_preds, sentence_preds, [x[0] for x in verified_results]], [doc_labels, sentence_labels, [x["label"] for x in data][:len(verified_results)]]):
            for name, metric in self.step2name2metrics[step].items():
                for metric_name, metric_value in metric.compute(predictions=preds, references=labels).items():
                    result[step][f"{name}_{metric_name}"] = str(metric_value)
        return result



