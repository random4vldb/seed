import collections
import datetime
from distutils.command.config import config
import json
import os
import random
import re
import time
from pathlib import Path
from tty import CFLAG

import hydra
import inflect
import jsonlines
import pandas as pd
import pyrootutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from elasticsearch_dsl import Q
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)
inflect = inflect.engine()


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim, labels):
        """Constructor
        Input: in_dim	- Dimension of input vector
                   out_dim	- Dimension of output vector
                   vocab	- Vocabulary of the embedding
        """
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.drop = torch.nn.Dropout(0.2)
        self.fc2 = nn.Linear(out_dim, labels)
        # self.soft_max = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        """Function for forward pass
        Input:	inp 	- Input to the network of dimension in_dim
        Output: output 	- Output of the network with dimension vocab
        """
        out_intermediate = F.relu(self.fc1(inp))
        output = self.fc2(out_intermediate)
        return output

def is_date(string):
    match = re.search('\d{4}-\d{2}-\d{2}', string)
    if match:
        try:
            date = datetime.datetime.strptime(match.group(), '%Y-%m-%d').date()
        except:
            return False
        return True
    else:
        return False


def load_sentences(file, skip_first=True, single_sentence=False):
    """Loads sentences into process-friendly format for a given file path.
    Inputs
    -------------------
    file    - str or pathlib.Path. The file path which needs to be processed
    skip_first      - bool. If True, skips the first line.
    single_sentence - bool. If True, Only the hypothesis statement is chosen.
                            Else, both the premise and hypothesis statements are
                            considered. This is useful for hypothesis bias experiments.

    Outputs
    --------------------
    rows    - List[dict]. Consists of all data samples. Each data sample is a
                    dictionary containing- uid, hypothesis, premise (except hypothesis
                    bias experiment), and the NLI label for the pair
    """
    rows = []
    df = pd.read_csv(file, sep="\t")
    for idx, row in df.iterrows():
        # Takes the relevant elements of the row necessary. Putting them in a dict,
        if single_sentence:
            sample = {
                "uid": row["annotator_id"],
                "hypothesis": row["hypothesis"],
                "label": int(row["label"]),
            }
        else:
            sample = {
                "uid": row["index"],
                "hypothesis": row["hypothesis"],
                "premise": row["premise"],
                "label": int(row["label"]),
            }

        rows.append(sample)  # Append the loaded sample
    return rows


def json_to_para(data, cfg):
    result = []

    if cfg.rand_perm == 2:
        table_ids = []
        for index, row in enumerate(data):
            table_ids += [row["table_id"]]
        random.shuffle(table_ids)
        for index, row in enumerate(data):
            row["table_id"] = table_ids[index]

    if cfg.rand_perm == 1:
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
        obj = {x:y if isinstance(y, list) else [y] for x,y in obj.items()}

        try:
            title = row["title"]
        except KeyError as e:
            print(row)
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
            key = key.replace(" A ", " AB ") # This is a hack to for inflect to work properly
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
                            "The " + str(key) + " of " + title + " " + verb_use + " "
                        )
                        line += (
                            "The " + str(key) + " of " + title + " " + verb_use + " "
                        )
                    except TypeError as e:
                        print(e)
                        print(row)
                        print(key)
                        print(title)
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


def preprocess_roberta(data, cfg):
    new_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_type)
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
        if cfg.single_sentence:
            encoded_inps = new_tokenizer(
                pt_dict["hypothesis"],
                padding="max_length",
                truncation=True,
                max_length=504,
            )
        else:
            encoded_inps = new_tokenizer(
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





def test(model, classifier, data, cfg):
    """Evaluate the model on a given dataset.
    Inputs
    ---------------
    model - transformers.AutoModel. The transformer model being used.
    classifier - torch.nn.Module. The classifier which sits on top of
                    the transformer model
    data - dict. Consists the processed input data

    Outputs
    ---------------
    accuracy - float. Accuracy of the model on that evaluation split
    gold_inds - List[int]. Gold labels
    predictions_ind - List[int]. Parallel list to gold_inds. Contains
                        label predictions
    """
    # Separate the data fields in the evaluation data
    enc = torch.tensor(data["encodings"]).cuda()
    attention_mask = torch.tensor(data["attention_mask"]).cuda()
    segs = torch.tensor(data["segments"]).cuda()
    labs = torch.tensor(data["labels"]).cuda()
    ids = torch.tensor(data["uid"]).cuda()

    # Create Data Loader for the split
    dataset = TensorDataset(enc, attention_mask, segs, labs, ids)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)

    model.eval()
    correct = 0
    total = 0
    gold_inds = []
    predictions_inds = []

    for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(loader):
        # Forward-pass w/o calculating gradients
        with torch.no_grad():
            outputs = model(enc, attention_mask=mask, token_type_ids=seg)
            # predictions = classifier(outputs[1])

        # Calculate metrics
        _, inds = torch.max(outputs.logits, 1)
        gold_inds.extend(gold.tolist())
        predictions_inds.extend(inds.tolist())
        correct += inds.eq(gold.view_as(inds)).cpu().sum().item()
        total += len(enc)

    accuracy = correct / total

    return accuracy, gold_inds, predictions_inds


def train_data(train_data, dev_data, test_data, cfg):
    """Train the transformer model on given data
    Inputs
    -------------
    args - dict. Arguments passed via CLI
    """

    # Creating required save directories
    if not os.path.isdir(cfg.output_dir):
        os.mkdir(cfg.output_dir)

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
    config = AutoConfig.from_pretrained(cfg.model_type)
    print(config.num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_type)
    embed_size = model.config.hidden_size
    classifier = FeedForward(
        embed_size, int(embed_size / 2), cfg.num_labels
    ).cuda()

    # Creating the training dataloaders
    dataset = TensorDataset(
        train_enc, train_attention_mask, train_segs, train_labs, train_ids
    )
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)


    # Intialize the optimizer and loss functions
    params = list(model.parameters())
    optimizer = optim.Adagrad(params, lr=0.0005)
    num_training_steps = len(loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps,
        )

    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    gradient_accumulation_steps = 2


    for ep in range(cfg.epochs):
        epoch_loss = 0
        start = time.time()
        # Iterate over batches

        for batch_ndx, (enc, mask, seg, gold, ids) in enumerate(tqdm(loader)):
            batch_loss = 0

            optimizer.zero_grad()
            # Forward-pass
            outputs = model(enc, attention_mask=mask, token_type_ids=seg, labels=gold)
            loss = outputs.loss / gradient_accumulation_steps

            accelerator.backward(loss)

            if (batch_ndx + 1) % gradient_accumulation_steps == 0:
                scheduler.step()
                optimizer.step()

            batch_loss += loss.item()
            epoch_loss += batch_loss

        normalized_epoch_loss = epoch_loss / (len(loader))
        print("Epoch {}".format(ep + 1))
        print("Epoch loss: {} ".format(normalized_epoch_loss))

        # Evaluate on the dev and test sets
        dev_acc, dev_gold, dev_pred = test(model, classifier, dev_data, cfg)
        test_acc, test_gold, test_pred = test(model, classifier, test_data, cfg)
        end = time.time()
        print("Dev Accuracy: {}".format(dev_acc))
        print("Time taken: {} seconds\n".format(end - start))
        # Save model
        torch.save(
            {
                "epoch": ep + 1,
                "model_state_dict": model.state_dict(),
                "classifier_state_dict": classifier.state_dict(),
                "loss": normalized_epoch_loss,
                "dev_accuracy": dev_acc,
            },
            os.path.join(cfg.output_dir, "model_"
            + str(ep + 1)
            + "_"
            + str(dev_acc))
        )
        json.dump({"epoch": ep + 1,"loss": normalized_epoch_loss, "dev_accuracy": dev_acc, "dev_gold": dev_gold, "dev_pred": dev_pred, 
        "test_accuracy": test_acc, "test_gold": test_gold, "test_pred": test_pred}, (Path(cfg.output_dir) / f"model_{ep + 1}.json").open("w"))
        logger.info({"epoch": ep + 1,"loss": normalized_epoch_loss, "dev_accuracy": dev_acc, "test_accuracy": test_acc})

        logger.info(f"Dev report: {classification_report(dev_gold, dev_pred, output_dict=True)}")
        logger.info(f"Test report: {classification_report(test_gold, test_pred, output_dict=True)}")


def test_data(data, cfg):
    """Test pre-trained model on evaluation splits
    Inputs
    ----------
    args - dict. Arguments passed via CLI
    """
    # Intialize model
    model = AutoModel.from_pretrained(cfg.model_type).cuda()
    embed_size = model.config.hidden_size
    classifier = FeedForward(embed_size, int(embed_size / 2), cfg.num_labels).cuda()

    # Load pre-trained models
    checkpoint = torch.load(os.path.join(cfg.output_dir, cfg.model_name))
    model.load_state_dict(checkpoint["model_state_dict"])
    classifier.load_state_dict(checkpoint["classifier_state_dict"])

    # Evaluate over splits

    # Compute Accuracy
    acc, gold, pred = test(model, classifier, data)

    results = {"accuracy": acc, "gold": gold, "pred": pred}

    return results


def read_infotab_file(file):
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
                "title": jobj["table_page_title"] if "table_page_title" in jobj else jobj["title"],
                "highlighted_cells": jobj["highlighted_cells"] if "highlighted_cells" in jobj else [],
            }
            idx += 1
            examples.append(example)
    print("Num examples", len(examples))
    return examples


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "infotab",
    config_name="sent_selection.yaml",
)
def main(cfg):
    # local_rank = torch.distributed.get_rank()

    if Path(cfg.cache_dir).exists() and (Path(cfg.cache_dir) / "train.json").exists() and cfg.cached:
        train_dataset = json.load((Path(cfg.cache_dir) / "train.json").open())
        if cfg.get("train_size", -1) != -1:
            train_dataset = {x: y[:cfg.train_size] for x, y in train_dataset.items()}
        dev_dataset = json.load((Path(cfg.cache_dir) / "dev.json").open()) 
        if cfg.get("dev_size", -1) != -1:
            dev_dataset = {x: y[:cfg.dev_size] for x, y in dev_dataset.items()}
    else:
        datasets = {}

        train_dataset, dev_dataset = read_infotab_file(cfg.train_file), read_infotab_file(cfg.dev_file)

        datasets = [train_dataset, dev_dataset]
        for idx in range(2):
            logger.info("Processing dataset ...")
            datasets[idx] = json_to_para(datasets[idx], cfg)
            datasets[idx] = preprocess_roberta(datasets[idx], cfg)

        logger.info("Saving preprocessed data ...")
        train_dataset, dev_dataset = datasets
        Path(cfg.cache_dir).mkdir(exist_ok=True)
        json.dump(train_dataset, (Path(cfg.cache_dir) / "train.json").open("w"))
        json.dump(dev_dataset, (Path(cfg.cache_dir) / "dev.json").open("w"))
    logger.info("Training")
    train_data(train_dataset, train_dataset, dev_dataset, cfg)


if __name__ == "__main__":
    main()
