import evaluate
import torch
from tango.integrations.pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


@LightningModule.register("seed::verification_model")

class Seed3Module(LightningModule):
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

        self.metrics = {
            "train_metrics": {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            },
            "val_metrics": {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            },
            "test_metrics": {
                "accuracy": evaluate.load("accuracy"),
                "precision": evaluate.load("precision"),
                "recall": evaluate.load("recall"),
                "f1": evaluate.load("f1"),
            },
        }

    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})

    def training_step(self, batch, batch_idx):
        outputs = self(
            **batch
        )

        self.log("train_loss", outputs.loss, on_step=True, on_epoch=False)
        return {"loss": outputs.loss, "preds": torch.argmax(outputs.logits, dim=1), "labels": batch["labels"]}

    def training_step_end(self, outputs):
        return outputs["loss"].mean()

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(
            **batch
        )

        self.log("val_loss", outputs.loss, on_step=True, on_epoch=False)
        return {"loss": outputs.loss, "preds": torch.argmax(outputs.logits, dim=1), "labels": batch["labels"]}


    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics["val_metrics"].items():
            self.log(
                f"val_{name}",
                metric.compute(predictions=preds, references=labels),
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
                f"test_{name}", metric.compute(predictions=preds, references=labels), prog_bar=True, on_epoch=True
            )

    def predict_step(self, batch, batch_idx):
        outputs = self(**batch)
        preds = torch.softmax(outputs.logits, dim=1)
        return preds

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
