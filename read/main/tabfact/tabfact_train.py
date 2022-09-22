# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import, division, print_function
from collections import OrderedDict
import argparse
import csv
import logging
import os
import random
from sched import scheduler
import sys
import io
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from accelerate import Accelerator
import hydra

from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    AdamW,
    BertConfig,
    get_linear_schedule_with_warmup,
)

from torch.utils.tensorboard import SummaryWriter
from pprint import pprint
import pyrootutils
logger = logging.getLogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            idx = 0
            for line in reader:
                idx += 1
                # if idx > 100: break
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, "utf-8") for cell in line)
                lines.append(line)
            return lines


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train"
        )

    def get_dev_examples(self, data_dir, dataset="dev"):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "{}.tsv".format(dataset))), dataset
        )

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                # column_types = [int(x) for x in line[2].split()]
                column_types = line[2].split()
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(
                (
                    InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label),
                    column_types,
                )
            )
        return examples


def convert_examples_to_features(
    examples,
    label_list,
    max_seq_length,
    tokenizer,
    output_mode,
    fact_place=None,
    balance=False,
    verbose=False,
):
    """Loads a data file into a list of `InputBatch`s."""
    assert fact_place is not None
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    pos_buf = []
    neg_buf = []
    logger.info("convert_examples_to_features ...")
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        example, column_types = example
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[: (max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # NOTE: fact is tokens_b and is now in front
        if fact_place == "first":
            tokens = ["[CLS]"] + tokens_b + ["[SEP]"]
            segment_ids = [0] * (len(tokens_b) + 2)

            assert len(tokens) == len(segment_ids)

            tokens += tokens_a + ["[SEP]"]
            segment_ids += [1] * (len(tokens_a) + 1)
        else:
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * (len(tokens_a) + 2)

            assert len(tokens) == len(segment_ids)

            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if verbose and ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        if balance:
            if label_id == 1:
                pos_buf.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                    )
                )
            else:
                neg_buf.append(
                    InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                    )
                )

            if len(pos_buf) > 0 and len(neg_buf) > 0:
                features.append(pos_buf.pop(0))
                features.append(neg_buf.pop(0))
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_id=label_id,
                )
            )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "qqp":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "tabfact",
    config_name="tabfact_train.yaml",
)
def main(cfg):

    processors = {
        "qqp": QqpProcessor,
    }

    output_modes = {
        "qqp": "classification",
    }

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if cfg.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.gradient_accumulation_steps
            )
        )

    cfg.train_batch_size = cfg.train_batch_size // cfg.gradient_accumulation_steps

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    if not cfg.get("train") and not cfg.get("eval"):
        raise ValueError("At least one of `train` or `eval` must be True.")

    logger.info(
        "Datasets are loaded from {}\n Outputs will be saved to {}".format(
            cfg.data_dir, cfg.output_dir
        )
    )
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
    writer = SummaryWriter(os.path.join(cfg.output_dir, "events"))

    task_name = cfg.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    output_mode = output_modes[task_name]

    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(
        cfg.model_name_or_path, do_lower_case=cfg.do_lower_case
    )

    train_examples = None
    num_train_optimization_steps = None
    if cfg.get("train"):
        train_examples = processor.get_train_examples(cfg.data_dir)
        num_train_optimization_steps = (
            int(
                len(train_examples)
                / cfg.train_batch_size
                / cfg.gradient_accumulation_steps
            )
            * cfg.num_train_epochs
        )


    model = BertForSequenceClassification.from_pretrained(
        cfg.model_name_or_path, num_labels=num_labels
    )

    accelerator = Accelerator()

    # Prepare optimizer
    if cfg.get("train"):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.learning_rate,
            correct_bias=False,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg.warmup_proportion * num_train_optimization_steps,
            num_training_steps=num_train_optimization_steps,
        )

    global_step = 0
    tr_loss = 0
    if cfg.get("train"):
        train_features = convert_examples_to_features(
            train_examples,
            label_list,
            cfg.max_seq_length,
            tokenizer,
            output_mode,
            fact_place=cfg.fact,
            balance=cfg.balance,
        )
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", cfg.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor(
            [f.input_ids for f in train_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in train_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in train_features], dtype=torch.long
        )

        if output_mode == "classification":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.long
            )
        elif output_mode == "regression":
            all_label_ids = torch.tensor(
                [f.label_id for f in train_features], dtype=torch.float
            )

        train_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )

        train_dataloader = DataLoader(
            train_data, batch_size=cfg.train_batch_size
        )

        model.train()
        train_dataloader, model, scheduler, optimizer = accelerator.prepare(train_dataloader, model, scheduler, optimizer)


        for epoch in trange(int(cfg.num_train_epochs), desc="Epoch"):
            logger.info("Training epoch {} ...".format(epoch))
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                # define a new function to compute loss values for both output_modes
                logits = model(input_ids, segment_ids, input_mask, labels=None)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if cfg.gradient_accumulation_steps > 1:
                    loss = loss / cfg.gradient_accumulation_steps

                accelerator.backward(loss)


                writer.add_scalar("train/loss", loss, global_step)
                tr_loss += loss.item()

                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % cfg.gradient_accumulation_steps == 0:
                    total_norm = 0.0
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1.0 / 2)
                    preds = torch.argmax(logits, -1) == label_ids
                    acc = torch.sum(preds).float() / preds.size(0)
                    writer.add_scalar("train/gradient_norm", total_norm, global_step)
                    if cfg.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = cfg.learning_rate * warmup_linear.get_lr(
                            global_step, cfg.warmup_proportion
                        )
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr_this_step
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    model.zero_grad()
                    global_step += 1

                if (step + 1) % cfg.period == 0:
                    # Save a trained model, configuration and tokenizer
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Only save the model it-self

                    # If we save using the predefined names, we can load using `from_pretrained`

                    output_dir = os.path.join(
                        cfg.output_dir, "save_step_{}".format(global_step)
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    model.eval()
                    torch.set_grad_enabled(False)  # turn off gradient tracking
                    evaluate(
                        cfg,
                        model,
                        device,
                        processor,
                        label_list,
                        num_labels,
                        tokenizer,
                        output_mode,
                        tr_loss,
                        global_step,
                        task_name,
                        tbwriter=writer,
                        save_dir=output_dir,
                    )
                    model.train()  # turn on train mode
                    torch.set_grad_enabled(True)  # start gradient tracking
                    tr_loss = 0

    # do eval before exit
    if cfg.get("eval"):
        if not cfg.get("train"):
            global_step = 0
            output_dir = None
        save_dir = output_dir if output_dir is not None else cfg.load_dir
        tbwriter = SummaryWriter(os.path.join(save_dir, "eval/events"))
        load_step = cfg.load_step
        if cfg.load_dir is not None:
            load_step = int(os.path.split(cfg.load_dir)[1].replace("save_step_", ""))
            print("load_step = {}".format(load_step))
        model.eval()
        evaluate(
            cfg,
            model,
            device,
            processor,
            label_list,
            num_labels,
            tokenizer,
            output_mode,
            tr_loss,
            global_step,
            task_name,
            tbwriter=tbwriter,
            save_dir=save_dir,
            load_step=load_step,
        )


def evaluate(
    cfg,
    model,
    device,
    processor,
    label_list,
    num_labels,
    tokenizer,
    output_mode,
    tr_loss,
    global_step,
    task_name,
    tbwriter=None,
    save_dir=None,
    load_step=0,
):

    if cfg.get("eval"):
        eval_examples = processor.get_dev_examples(cfg.data_dir, dataset=cfg.test_set)
        eval_features = convert_examples_to_features(
            eval_examples,
            label_list,
            cfg.max_seq_length,
            tokenizer,
            output_mode,
            fact_place=cfg.fact,
            balance=False,
        )
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", cfg.eval_batch_size)
        all_input_ids = torch.tensor(
            [f.input_ids for f in eval_features], dtype=torch.long
        )
        all_input_mask = torch.tensor(
            [f.input_mask for f in eval_features], dtype=torch.long
        )
        all_segment_ids = torch.tensor(
            [f.segment_ids for f in eval_features], dtype=torch.long
        )

        if output_mode == "classification":
            all_label_ids = torch.tensor(
                [f.label_id for f in eval_features], dtype=torch.long
            )
        elif output_mode == "regression":
            all_label_ids = torch.tensor(
                [f.label_id for f in eval_features], dtype=torch.float
            )

        eval_data = TensorDataset(
            all_input_ids, all_input_mask, all_segment_ids, all_label_ids
        )
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=cfg.eval_batch_size
        )

        batch_idx = 0
        eval_loss = 0
        nb_eval_steps = 0
        preds = []
        temp = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(
            eval_dataloader, desc="Evaluating"
        ):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None)

            # create eval loss and other metric required by the task
            if output_mode == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(
                    logits.view(-1, num_labels), label_ids.view(-1)
                )
            elif output_mode == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
            else:
                preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis=0)

            labels = label_ids.detach().cpu().numpy().tolist()
            start = batch_idx * cfg.eval_batch_size
            end = start + len(labels)
            batch_range = list(range(start, end))
            csv_names = [
                eval_examples[i][0].guid.replace("{}-".format(cfg.test_set), "")
                for i in batch_range
            ]
            facts = [eval_examples[i][0].text_b for i in batch_range]
            labels = label_ids.detach().cpu().numpy().tolist()
            assert len(csv_names) == len(facts) == len(labels)

            temp.extend([(x, y, z) for x, y, z in zip(csv_names, facts, labels)])
            batch_idx += 1

        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        if output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif output_mode == "regression":
            preds = np.squeeze(preds)

        evaluation_results = OrderedDict()
        for x, y in zip(temp, preds):
            c, f, l = x
            if not c in evaluation_results:
                evaluation_results[c] = [{"fact": f, "gold": int(l), "pred": int(y)}]
            else:
                evaluation_results[c].append(
                    {"fact": f, "gold": int(l), "pred": int(y)}
                )

        print("save_dir is {}".format(save_dir))
        output_eval_file = os.path.join(
            save_dir, "{}_eval_results.json".format(cfg.test_set)
        )
        with io.open(output_eval_file, "w", encoding="utf-8") as fout:
            json.dump(evaluation_results, fout, sort_keys=True, indent=4)

        result = compute_metrics(task_name, preds, all_label_ids.numpy())
        loss = tr_loss / cfg.period if cfg.get("train") and global_step > 0 else None

        log_step = global_step if cfg.get("train") and global_step > 0 else load_step
        result["eval_loss"] = eval_loss
        result["global_step"] = log_step
        result["loss"] = loss

        output_eval_metrics = os.path.join(save_dir, "eval_metrics.txt")
        with open(output_eval_metrics, "a") as writer:
            logger.info("***** Eval results {}*****".format(cfg.test_set))
            writer.write("***** Eval results {}*****\n".format(cfg.test_set))
            for key in sorted(result.keys()):
                if result[key] is not None and tbwriter is not None:
                    tbwriter.add_scalar(
                        "{}/{}".format(cfg.test_set, key), result[key], log_step
                    )
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == "__main__":
    main()
