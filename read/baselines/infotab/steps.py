import collections
import json
import os
import random

import evaluate
import inflect
import jsonlines
import pandas as pd
import pyrootutils
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from loguru import logger
from tango import JsonFormat, Step, Format, TorchFormat
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from .classifier import FeedForward
from .helpers import *


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)
inflect = inflect.engine()


@Step.register("infotab_json_to_para")
class InfoTabJsonToPara(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True
    FORMAT = JsonFormat()

    def run(self, data, rand_perm, ):
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
                                "Error in key: %s in article of title %s", key, title
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
                        para += title + " was " + str(key) + " on " + values[0] + ". "
                        line += title + " was " + str(key) + " on " + values[0] + ". "
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
        return result


@Step.register("infotab_preprocess")
class InfoTabPreprocess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True

    def run(self, data, tokenizer, single_sentence):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        # Process for every split
        # Initialize dictionary to store processed information
        keys = ["uid", "encodings", "attention_mask", "segments", "labels"]
        data_dict = {key: [] for key in keys}
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
                )
            else:
                encoded_inps = tokenizer(
                    pt_dict["premise"],
                    pt_dict["hypothesis"],
                    padding="max_length",
                    truncation=True,
                    max_length=504,
                )

            # Some models do not return token_type_ids and hence
            # we just return a list of zeros for them. This is just
            # required for completeness.
            if "token_type_ids" not in encoded_inps.keys():
                encoded_inps["token_type_ids"] = [0] * len(encoded_inps["input_ids"])

            data_dict["uid"].append(int(pt_dict["index"]))
            data_dict["encodings"].append(encoded_inps["input_ids"])
            data_dict["attention_mask"].append(encoded_inps["attention_mask"])
            data_dict["segments"].append(encoded_inps["token_type_ids"])
            data_dict["labels"].append(pt_dict["label"])

            if (samples_processed % 100) == 0:
                print("{} examples processed".format(samples_processed))

        print("Preprocessing Finished")
        return data_dict


@Step.register("infotab_train")
class InfoTabTrain(Step):
    DETERMINISTIC: bool = False
    CACHEABLE = False
    FORMAT: Format = JsonFormat()

    def run(self, train_data, dev_data, test_data, model_name_or_path, batch_size, num_epochs, output_dir):
        """Train the transformer model on given data
        Inputs
        -------------
        args - dict. Arguments passed via CLI
        """

        # Creating required save directories
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)


        print("{} train data loaded".format(len(train_data["encodings"])))
        print("{} dev data loaded".format(len(dev_data["encodings"])))
        print("{} test data loaded".format(len(test_data["encodings"])))

        # Separating the data fields
        train_enc = torch.tensor(train_data["encodings"]).cuda()
        train_attention_mask = torch.tensor(train_data["attention_mask"]).cuda()
        train_segs = torch.tensor(train_data["segments"]).cuda()
        train_labs = torch.tensor(train_data["labels"]).cuda().long()
        train_ids = torch.tensor(train_data["uid"]).cuda()

        accelerator = Accelerator()

        # Intialize Models
        config = AutoConfig.from_pretrained(model_name_or_path)
        model = AutoModel.from_pretrained(model_name_or_path)
        embed_size = model.config.hidden_size
        classifier = FeedForward(embed_size, int(embed_size / 2), 2).cuda()

        # Creating the training dataloaders
        dataset = TensorDataset(
            train_enc, train_attention_mask, train_segs, train_labs, train_ids
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Intialize the optimizer and loss functions
        params = list(model.parameters())
        optimizer = optim.Adagrad(params, lr=0.0001)
        loss_fn = nn.CrossEntropyLoss()
        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
        model.train()

        gradient_accumulation_steps = 2

        for ep in range(num_epochs):
            epoch_loss = 0
            # Iterate over batches

            for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(tqdm(loader)):
                batch_loss = 0

                optimizer.zero_grad()
                # Forward-pass
                outputs = model(enc, attention_mask=mask, token_type_ids=seg)
                predictions = classifier(outputs[1])

                loss = loss_fn(predictions, gold)

                accelerator.backward(loss)

                if (batch_ndx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()

                batch_loss += loss.item()
                epoch_loss += batch_loss

            normalized_epoch_loss = epoch_loss / (len(loader))

            # Evaluate on the dev and test sets
            if dev_data:
                accelerator.print("Evaluating on dev set")
                self.test(model, classifier, dev_data, batch_size)

            # Save model
            unwrapped_model = accelerator.unwrap_model(model)

            torch.save(
                {
                    "epoch": ep + 1,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "classifier_state_dict": classifier.state_dict(),
                    "loss": normalized_epoch_loss,
                },
                os.path.join(output_dir, "model_" + str(ep + 1)),
            )

        if test_data:
            accelerator.print("Evaluating on test set")
            self.test(model, classifier, test_data, batch_size)

        unwrapped_model = accelerator.unwrap_model(model)
        return {"model": unwrapped_model, "classifier": classifier}

    def test(self, model, classifier, data, batch_size):
        # Separate the data fields in the evaluation data
        enc = torch.tensor(data["encodings"]).cuda()
        attention_mask = torch.tensor(data["attention_mask"]).cuda()
        segs = torch.tensor(data["segments"]).cuda()
        labs = torch.tensor(data["labels"]).cuda()
        ids = torch.tensor(data["uid"]).cuda()

        # Create Data Loader for the split
        dataset = TensorDataset(enc, attention_mask, segs, labs, ids)
        loader = DataLoader(dataset, batch_size=batch_size)

        model.eval()
        gold_inds = []
        predictions_inds = []

        accelerator = Accelerator()
        model, classifier, loader = accelerator.prepare(model, classifier, loader)
        metrics = {
            k: evaluate.load(k) for k in ["accuracy", "f1", "precision", "recall"]
        }

        for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(loader):
            with torch.no_grad():
                outputs = model(enc, attention_mask=mask, token_type_ids=seg)
                predictions = classifier(outputs[1])

            # Calculate metrics
            _, inds = torch.max(predictions, 1)
            for metric_name, metric in metrics.items():
                metric.add_batch(
                    predictions=accelerator.gather_for_metrics(inds),
                    references=accelerator.gather_for_metrics(gold),
                )



        for metric_name, metric in metrics.items():
            accelerator.print(f"{metric_name}: {metric.compute()}")

        return gold_inds, predictions_inds


    def test_data(model, data, cfg):
        # Intialize model
        accelerator = Accelerator()
        model = AutoModel.from_pretrained(cfg.model_type).cuda()
        embed_size = model.config.hidden_size
        classifier = FeedForward(embed_size, int(embed_size / 2), cfg.num_labels).cuda()

        # Load pre-trained models
        checkpoint = torch.load(os.path.join(cfg.output_dir, cfg.model_name))

        model.load_state_dict(checkpoint["model_state_dict"])
        classifier.load_state_dict(checkpoint["classifier_state_dict"])

        # Evaluate over splits

        # Compute Accuracy
        acc, gold, pred = self.test(model, classifier, data, cfg)

        results = {"accuracy": acc, "gold": gold, "pred": pred}

        metrics = {k: evaluate.load(k) for k in ["accuracy", "f1", "precision", "recall"]}

        for metric_name, metric in metrics.items():
            results[metric_name] = metric.compute(predictions=gold, references=pred)
            logger.info(f"{metric_name}: {results[metric_name]}")

        return results


@Step.register("infotab_input_data")
class InfoTabInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE = True

    def run(self, file):
        examples = []
        with jsonlines.open(file, "r") as reader:
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
                    if "highlighted_cells" in jobj
                    else [],
                }
                idx += 1
                examples.append(example)
        logger.info("Num examples", len(examples))
        return examples
