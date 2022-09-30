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
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.nn import ModuleDict
from tango.integrations.pytorch_lightning import LightningModule


@LightningModule.register("seed_sent_selection")
class SentSelectModule(LightningModule):
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

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=2)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=self.config, ignore_mismatched_sizes=True
        )

        self.metrics = ModuleDict(
            [
                [
                    "train_metrics",
                    ModuleDict(
                        [
                            [metric.__class__.__name__, metric]
                            for metric in [Accuracy(), Precision(), Recall(), F1Score()]
                        ]
                    ),
                ],
                [
                    "val_metrics",
                    ModuleDict(
                        [
                            [metric.__class__.__name__, metric]
                            for metric in [Accuracy(), Precision(), Recall(), F1Score()]
                        ]
                    ),
                ],
            ]
        )

    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        preds = torch.log_softmax(outputs.logits, dim=1)
        loss = F.nll_loss(preds, batch["labels"])
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return {
            "loss": loss,
            "preds": torch.argmax(preds, dim=1),
            "labels": batch["labels"],
        }

    def training_step_end(self, outputs):
        preds = outputs["preds"]
        labels = outputs["labels"]
        for name, metric in self.metrics["train_metrics"].items():
            self.log(
                f"train_{name}",
                metric(preds, labels),
                prog_bar=True,
            )

        return outputs["loss"].sum()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)

        preds = torch.log_softmax(outputs.logits, dim=1)
        loss = F.nll_loss(preds, batch["labels"])
        self.log("val_loss", loss, on_step=True, on_epoch=False)
        return {
            "loss": loss,
            "preds": torch.argmax(preds, dim=1),
            "labels": batch["labels"],
        }

    def validation_epoch_end(self, outputs):
        loss = torch.cat([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics["val_metrics"].items():
            self.log(
                f"val_{name}",
                metric(preds, labels),
                prog_bar=True,
                on_epoch=True,
            )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        return {"preds": preds, "labels": batch["labels"]}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        for name, metric in self.metrics["test_metrics"].items():
            self.log(
                f"test_{name}", metric(preds, labels), prog_bar=True, on_epoch=True
            )

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
