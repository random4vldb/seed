import json
from pathlib import Path

import hydra
import pandas as pd
import pyrootutils
import torch
from accelerate import Accelerator
from loguru import logger
from torchmetrics import Accuracy, F1Score, Precision, Recall
from transformers import (
    TapasForSequenceClassification,
    TapasTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        ex_id = idx % 2
        idx = idx // 2
        item = self.df.iloc[idx]
        if ex_id == 0:
            table = pd.DataFrame(json.loads(item["positive"]))
        else:
            table = pd.DataFrame(json.loads(item["negative"]))
        cells = zip(*item["highlighted_cells"])
        cells = [list(x) for x in cells]
        sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)


        encoding = self.tokenizer(table=sub_table, 
                                queries=item["sentence"],
                                padding="max_length",
                                truncation=True,
                                max_length=512,
                                return_tensors="pt")
        if ex_id == 0:
            encoding["label"] = torch.tensor([1])
        else:
            encoding["label"] = torch.tensor([0])

        encoding = {key: val[-1] for key, val in encoding.items()}
        return encoding

    def __len__(self):
        return len(self.df)


@hydra.main(version_base="1.2", config_path=root / "config" / "tapas", config_name="tapas_sent_train.yaml")
def main(cfg):
    tokenizer = TapasTokenizer.from_pretrained("google/tapas-base")
    model = TapasForSequenceClassification.from_pretrained("google/tapas-base")

    train_df = pd.read_json(cfg.train_file, lines=True)
    dev_df = pd.read_json(cfg.dev_file, lines=True)

    train_dataset = TableDataset(train_df, tokenizer)
    dev_dataset = TableDataset(dev_df, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, collate_fn=data_collator)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=data_collator)

    accelerate = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=0, num_training_steps=cfg.epochs * len(train_dataloader))


    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    model, optimizer, train_dataloader, dev_dataloader, scheduler = accelerate.prepare(
        model, optimizer, train_dataloader, dev_dataloader, scheduler
    )

    name2torchmetric = {
        "accuracy": Accuracy(), 
        "precision": Precision(), 
        "recall": Recall(), 
        "f1": F1Score()
    }


    for epoch in range(cfg.epochs):
        model.train()
        logger.info(f"Epoch {epoch}")
        for batch in tqdm(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerate.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        model.eval()
        for batch in dev_dataloader:
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            for name, metric in name2torchmetric.items():
                metric(logits, labels)

        for name, metric in name2torchmetric.items():
            logger.info(f"{name}: {metric.compute()}")
            metric.reset()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
