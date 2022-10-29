import collections
import datetime
from typing import Optional

import jaro
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.special import softmax
from tango import Format, JsonFormat, Step
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    pipeline,
)
from torchmetrics import (
    Precision,
    Recall,
    F1Score,
    RetrievalPrecision,
    RetrievalRecall,
    RetrievalHitRate,
    RetrievalMRR,
    Accuracy
)
import json
import torch


def tokenize(tokenizer, tokenizer_type, x):
    if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
        table = pd.DataFrame(json.loads(x["table"])).astype(str)
        inputs = tokenizer(
            table, x["sent"], padding=True, truncation=True, return_tensors="pt"
        )
    else:
        table = x["table"]
        inputs = tokenizer(
            [(table, x["sent"])], padding=True, truncation=True, return_tensors="pt"
        )

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    return inputs


@Step.register("pipeline::sentence_selection")
class SentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "00672"

    def run(self, model, tokenizer, data, doc_results, batch_size):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

        tables = []
        sentences = []
        ids = []

        all_scores = []
        all_preds = []

        collator = DataCollatorWithPadding(tokenizer)

        training_args = TrainingArguments(
            output_dir="models/sentence_selection",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            remove_unused_columns=False,
            label_names=["label"],
            eval_accumulation_steps=2,
        )

        trainer = Trainer(
            args=training_args,
            model=model,
            data_collator=collator,
        )

        for idx, (example, doc_result) in enumerate(list(zip(data, doc_results))):
            for sent, score, title in doc_result:
                if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
                    tables.append(json.dumps(example["table"]))
                else:
                    tables.append(example["linearized_table"])
                sentences.append(sent)
                print(sent)
                ids.append(idx)
            print("----------------------------------------------------------------------")


        assert max(ids) == len(data) - 1
        for i in range(0, len(tables), 10000):
            with training_args.main_process_first(desc="dataset map pre-processing"):
                dataset = Dataset.from_dict(
                    {"table": tables[i : i + 10000], "sent": sentences[i : i + 10000]}
                )
                dataset = dataset.map(
                    lambda x: tokenize(tokenizer, tokenizer_type, x),
                    batch_size=batch_size,
                    remove_columns=["table", "sent"],
                )

            outputs = trainer.predict(
                dataset, ignore_keys=["past_key_values", "encoder_last_hidden_state"]
            ).predictions

            preds = np.argmax(outputs, axis=1).tolist()
            scores = softmax(outputs, axis=1)[:, 1].tolist()

            all_preds.extend(preds)
            all_scores.extend(scores)

        assert len(ids) == len(all_preds) == len(all_scores) == len(sentences)
        sentence_results = [[[], [], []] for _ in range(max(ids) + 1)]
        for (pred, sent, score, id) in zip(all_preds, sentences, all_scores, ids):
            if pred == 1:
                sentence_results[id][0].append((sent, score))
            sentence_results[id][1].append((sent, pred))
            sentence_results[id][2] = (
                data[id]["linearized_table"],
                data[id]["sentence"],
            )
        return sentence_results


@Step.register("pipeline::table_verification")
class TableVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0061"

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
                    tables.append(json.dumps(example["table"]))
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
            eval_accumulation_steps=4,
        )

        dataset = Dataset.from_dict({"table": tables, "sent": sentences})
        dataset = dataset.map(
            lambda x: tokenize(tokenizer, tokenizer_type, x),
            batch_size=32,
            remove_columns=["table", "sent"],
            keep_in_memory=True,
        )
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


@Step.register("pipeline::cell_correction")
class CellCorrection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "002"

    def choose_question(self, table, column):
        if (
            "date" in column.lower()
            or "year" in column.lower()
            or "month" in column.lower()
        ):
            return "When ?"
        if "time" in column.lower():
            return "What time ?"
        try:
            datetime.strptime(table[column][0])
            return "When ?"
        except:
            pass

        return "What " + column + "?"

    def compare(self, value1, value2):
        ordinals = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]

        for ordinal in ordinals:
            if ordinal in value1:
                value1 = value1.replace(ordinal, str(ordinals.index(ordinal) + 1))
            if ordinal in value2:
                value2 = value2.replace(ordinal, str(ordinals.index(ordinal) + 1))

        if jaro.jaro_winkler_metric(value1, value2) > 0.8:
            return True

    def run(self, data, verified_results, batch_size=8):
        model_name = "deepset/roberta-base-squad2"

        # a) Get predictions
        nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)

        results = []
        for i, (example, result) in enumerate(zip(data, verified_results)):
            table = pd.DataFrame(example["table"]).astype(str)
            cells = zip(*example["highlighted_cells"])
            cells = [list(x) for x in cells]
            sub_table = (
                table.iloc[cells[0], cells[1]]
                .drop_duplicates()
                .reset_index()
                .astype(str)
            )
            sub_table = sub_table.iloc[:, ~sub_table.columns.duplicated()]

            verified_result, sents = result
            if verified_result == 1:
                results.append(None)
                continue
            for column in sub_table.columns:
                if column == "index":
                    continue
                for sent in sents:
                    question = self.choose_question(sub_table, column)
                    # input_string = question + "\\n" + sent[0]
                    answer = nlp({"question": question, "context": sent[0]})

                    if (
                        jaro.jaro_winkler_metric(
                            answer["answer"], sub_table.loc[0, column]
                        )
                        > 0.8
                    ):
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
    VERSION: Optional[str] = "00585"

    step2name2metrics = {
        x: {
            "accuracy": Accuracy(),
            "f1": F1Score(),
            "precision": Precision(),
            "recall": Recall(),
        }
        for x in ["table_verification", "cell_correction"]
    }

    step2name2metrics["sentence_selection"] = {
        "accuracy": RetrievalHitRate(),
        "recall": RetrievalRecall(),
        "precision": RetrievalPrecision(),
        "mrr": RetrievalMRR(),
    }

    step2name2metrics["document_retrieval"] = {
        "accuracy": RetrievalHitRate(),
        "recall": RetrievalRecall(),
        "precision": RetrievalPrecision(),
        "mrr": RetrievalMRR(),
    }



    def jaccard_similarity(self, list1, list2):
        return len(set(list1).intersection(set(list2))) / len(
            set(list1).union(set(list2))
        )

    def process_cell_correction(self, data, correction_results):
        preds = []
        labels = []
        for i, (example, result) in enumerate(zip(data, correction_results)):
            if "negatives" not in example or example["negatives"] == "":
                labels.append(None)
            else:
                (
                    replacing_value,
                    replaced_value,
                    replaced_row,
                    valid_column,
                ) = json.loads(example["negatives"])
                labels.append(replaced_value)

            preds.append(result)
        return preds, labels

    def process_sentence_selection(self, data, sentence_results):
        preds = []
        labels = []
        indices = []
        for idx, (example, result) in enumerate(zip(data, sentence_results)):
            for sent, pred in result:
                if pred == 1:
                    if (
                        self.jaccard_similarity(sent.split(), example["sentence"].split())
                        > 0.7
                    ):
                        labels.append(1)
                    else:
                        labels.append(0)
                    preds.append(1.0)
                    indices.append(idx)
        return preds, labels, indices

    def process_document_retrieval(self, data, doc_results, align_title=False):
        preds = []
        labels = []
        indices = []
        for idx, (example, result) in enumerate(zip(data, doc_results)):
            if align_title:
                example_title = example["title"][: example["title"].find("(") - 1]
            else:
                example_title = example["title"]

            for rank, (doc, score, title) in enumerate(
                sorted(result, key=lambda x: x[1], reverse=True)
            ):

                if title == example_title:
                    labels.append(1)
                    break
            else:
                labels.append(0)
            preds.append(1.0)
            indices.append(idx)

        return preds, labels, indices

    def run(self, data, doc_results, sentence_results, verified_results, align_title=False):
        sentence_results = [x[1] for x in sentence_results]
        doc_preds, doc_labels, doc_indices = self.process_document_retrieval(data, doc_results, align_title=align_title)
        sentence_preds, sentence_labels, sentence_indices = self.process_sentence_selection(
            data, sentence_results
        )
        cell_preds, cell_labels = self.process_cell_correction(data, verified_results)

        result = collections.defaultdict(dict)
        step2failed_cases = collections.defaultdict(list)

        labels = [x["label"] for x in data]

        if max(labels) == 2:
            labels = [1 if x == 2 else x for x in labels]

        for step, preds, labels, indices, errors in zip(
            [
                "document_retrieval",
                "sentence_selection",
                "table_verification",
            ],
            [doc_preds, sentence_preds, [x[0] for x in verified_results], cell_preds],
            [
                doc_labels,
                sentence_labels,
                labels,
                cell_labels,
            ],
            [doc_indices, sentence_indices, None, None],
            [doc_results, sentence_results, verified_results, None],
        ):
            for name, metric in self.step2name2metrics[step].items():
                if indices is not None:
                    result[step][f"{name}"] = metric(torch.tensor(preds).float(), torch.tensor(labels), indexes=torch.tensor(indices))
                    for pred, label, index in zip(preds, labels, indices):
                        if pred != label:
                            step2failed_cases[step].append((data[index], errors[index]))
                else:
                    result[step][f"{name}"] = metric(torch.tensor(preds).float(), torch.tensor(labels))


        return {"result": result, "failed_cases": step2failed_cases}
