import json
from multiprocessing.context import ForkContext

import evaluate
import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from read.metrics.generation import Table2TextEvaluator
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import pyrootutils
from loguru import logger
from sacremoses import MosesDetokenizer
import six


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

detokenizer = MosesDetokenizer(lang="en")
args = Seq2SeqTrainingArguments(
    output_dir="temp/totto/model_test",
    num_train_epochs=10,
    predict_with_generate=True,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-5,
    eval_accumulation_steps=2
)

tokenizer = T5Tokenizer.from_pretrained("t5-small", model_max_length=512)
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("temp/totto/model")

logger.info(f"Finished loading model and tokenizer")

datasets = DatasetDict()

df = pd.read_json("data/totto/standard/train.jsonl", lines=True)[:1]
df["sentence"] = df["sentence_annotations"].apply(
    lambda x: [ann["final_sentence"] for ann in x][0]
)
# df = df.explode("sentence").reset_index(drop=True)
datasets["train"] = Dataset.from_pandas(df[["subtable_metadata_str", "sentence"]])


df = pd.read_json("data/totto/standard/dev.jsonl", lines=True)
df["sentence"] = df["sentence_annotations"].apply(
    lambda x: [ann["final_sentence"] for ann in x][0]
)
datasets["test"] = Dataset.from_pandas(df[["subtable_metadata_str", "sentence"]])

logger.info(f"Finished loading datasets")

def tokenize(batch):
    encodings = tokenizer(batch["subtable_metadata_str"], truncation=True, padding=True)
    encodings["labels"] = tokenizer(batch["sentence"], truncation=True, padding=True)[
        "input_ids"
    ]
    return encodings


tokenized_datasets = datasets.map(
    tokenize,
    batched=True,
    batch_size=32,
    remove_columns=["subtable_metadata_str", "sentence"]
)

logger.info("Finished tokenizing datasets")

train_dataset, test_dataset = tokenized_datasets["train"], tokenized_datasets["test"]

data_collator = DataCollatorForSeq2Seq(tokenizer, padding=True)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# trainer.train()
# trainer.evaluate(train_dataset, metric_key_prefix="train_eval", num_beams=5, max_length=128)
pred = trainer.predict(test_dataset, num_beams=4, max_length=128)

bleu = evaluate.load("sacrebleu")
pred_ids = pred.predictions
pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
pred_str = list(map(str.strip, pred_str))
references = []

for annotation in df["sentence_annotations"].values.tolist():
    sub_references = []
    for i in range(3):
        if i < len(annotation):
            sub_references.append(annotation[i]["final_sentence"].strip())
        else:
            sub_references.append("")
    references.append(sub_references)

# trainer.save_model("temp/totto/model")

with open("temp/totto/pred.txt", "w") as f:
    for pred in pred_str:
        f.write(pred + "\n")

for i in range(3):
    with open(f"temp/totto/ref{i}.txt", "w") as f:
        for ref in references:
            f.write(ref[i] + "\n")
