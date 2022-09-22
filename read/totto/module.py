from collections import defaultdict
import torch
from pytorch_lightning import (
    LightningModule,
)
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from yaml import load
import torchmetrics
import torch
import torch.nn.functional as F


class Seed2Module(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        margin: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(
            model_name_or_path, num_labels=2
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config, ignore_mismatched_sizes=True
        )

        self.metrics = defaultdict(dict)

        self.train_acc = torchmetrics.Accuracy(num_labels=2)
        self.metrics["train"] = {"acc": self.train_acc}

        self.val_acc = torchmetrics.Accuracy(num_labels=2)
        self.metrics["val"] = {"acc": self.val_acc}

        self.test_acc = torchmetrics.Accuracy(num_labels=2)
        self.metrics["test"] = {"acc": self.test_acc}

    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})
        

    def training_step(self, batch, batch_idx):
        outputs1 = self(**{k.replace("positive_", ""): v for k, v in batch.items() if "positive_" in k})
        outputs2 = self(**{k.replace("negative_", ""): v for k, v in batch.items() if "negative_" in k})
        preds = torch.softmax(outputs1.logits, dim=1)[:, 0] - torch.softmax(outputs2.logits, dim=1)[:, 0]
        loss = F.relu(preds + self.hparams.margin).mean()        
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return {"loss": loss, "preds": preds}

    def training_step_end(self, outputs):
        preds = outputs["preds"]
        for name, metric in self.metrics["train"].items():
            self.log(f"train_{name}", metric(preds < 0, torch.ones_like(preds).long()), prog_bar=True)

        return outputs['loss'].sum()


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs1 = self(**{k.replace("positive_", ""): v for k, v in batch.items() if "positive_" in k})
        outputs2 = self(**{k.replace("negative_", ""): v for k, v in batch.items() if "negative_" in k})
        preds = torch.softmax(outputs1.logits, dim=1)[:, 0] - torch.softmax(outputs2.logits, dim=1)[:, 0] 
        loss = F.relu(preds + self.hparams.margin).mean()     
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        return {"loss": loss, "preds": preds}

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics["val"].items():
            self.log(f"val_{name}", metric(preds < 0, torch.ones_like(preds).long()), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits)
        return {"preds": preds, "labels": batch["labels"]}


    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat(x["labels"] for x in outputs)
        for name, metric in self.metrics["test"].items():
            self.log(f"test_{name}", metric(preds, labels), prog_bar=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
