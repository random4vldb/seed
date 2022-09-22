import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchmetrics
from loguru import logger
from pytorch_lightning import LightningModule
from transformers import (
    AdamW,
    DPRContextEncoder,
    DPRQuestionEncoder,
    get_linear_schedule_with_warmup,
    DPRQuestionEncoderTokenizerFast,
    DPRContextEncoderTokenizerFast,
)
from pathlib import Path
from typing import Union

class PooledEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy):
        pooler_output = self.encoder(input_ids, attention_mask)[0]
        return pooler_output

class BiEncoder(torch.nn.Module):
    """
    This trains the DPR encoders to maximize dot product between queries and positive contexts.
    We only use this model during training.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qry_model = PooledEncoder(DPRQuestionEncoder.from_pretrained(cfg.qry_encoder_name_or_path))
        self.ctx_model = PooledEncoder(DPRContextEncoder.from_pretrained(cfg.ctx_encoder_name_or_path))
        self.saved_debug = False

    def encode(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # if 0 < self.cfg.encoder_gpu_train_limit:
        #     # checkpointing
        #     # dummy requries_grad to deal with checkpointing issue:
        #     #   https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/13
        #     dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
        #     all_pooled_output = []
        #     for sub_bndx in range(0, input_ids.shape[0], self.cfg.encoder_gpu_train_limit):
        #         sub_input_ids = input_ids[sub_bndx:sub_bndx+self.cfg.encoder_gpu_train_limit]
        #         sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.cfg.encoder_gpu_train_limit]
        #         pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
        #         all_pooled_output.append(pooler_output)
        #     return torch.cat(all_pooled_output, dim=0)
        # else:
        return model(input_ids, attention_mask, None)

    def save_for_debug(self, qry_reps, ctx_reps, positive_indices):
        if self.cfg.global_rank == 0 and not self.saved_debug and \
                self.cfg.debug_location and not os.path.exists(self.cfg.debug_location):
            os.makedirs(self.cfg.debug_location)
            torch.save(qry_reps, os.path.join(self.cfg.debug_location, 'qry_reps.bin'))
            torch.save(ctx_reps, os.path.join(self.cfg.debug_location, 'ctx_reps.bin'))
            # torch.save(positive_indices, os.path.join(self.cfg.debug_location, 'positive_indices.bin'))
            self.saved_debug = True
            logger.warning(f'saved debug info at {self.cfg.debug_location}')

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        positive_input_ids: torch.Tensor,
        positive_attention_mask: torch.Tensor,
        negative_input_ids: torch.Tensor,
        negative_attention_mask: torch.Tensor,
    ):
        """
        All batches must be the same size (q and c are fixed during training)
        :param input_ids_q: q x seq_len_q [0, vocab_size)
        :param attention_mask_q: q x seq_len_q [0, 1]
        :param input_ids_c: c x seq_len_c
        :param attention_mask_c: c x seq_len_c
        :param positive_indices: q [0, c)
        :return:
        """
        qry_reps = self.encode(self.qry_model, query_input_ids, query_attention_mask)
        input_ids_c = torch.cat([positive_input_ids, negative_input_ids], dim=0)
        attention_mask_c = torch.cat([positive_attention_mask, negative_attention_mask], dim=0)
        ctx_reps = self.encode(self.ctx_model, input_ids_c, attention_mask_c)
        dot_products = torch.matmul(qry_reps, ctx_reps.transpose(0, 1))  # (q * world_size) x (c * world_size)
        probs = F.log_softmax(dot_products, dim=1)
        labels = torch.arange(0, qry_reps.shape[0]).type_as(probs).long()
        loss = F.nll_loss(probs, labels)
        predictions = torch.max(probs, 1)[1]
        return loss, predictions, labels

    def save(self, save_dir: Union[str, os.PathLike]):
        self.qry_model.encoder.save_pretrained(os.path.join(save_dir, 'qry_encoder'))
        self.ctx_model.encoder.save_pretrained(os.path.join(save_dir, 'ctx_encoder'))


class DPRModule(LightningModule):
    def __init__(
        self,
        num_labels: int,
        qry_encoder_name_or_path: str,
        ctx_encoder_name_or_path: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        margin: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = BiEncoder(self.hparams)
        self.metrics = defaultdict(dict)

        self.train_acc = torchmetrics.Accuracy()
        self.metrics["train"] = {"acc": self.train_acc}

        self.val_acc = torchmetrics.Accuracy()
        self.metrics["val"] = {"acc": self.val_acc}

        self.test_acc = torchmetrics.Accuracy()
        self.metrics["test"] = {"acc": self.test_acc}

    def forward(self, **inputs):
        return self.model(**{k: v.long() for k, v in inputs.items()})


    def save(self, output_path):
        output_path = Path(output_path)
        self.model.ctx_model.encoder.save_pretrained(output_path / "ctx_encoder_dpr")
        self.model.qry_model.encoder.save_pretrained(output_path / "qry_encoder_dpr")
        tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained("facebook/dpr-question_encoder-multiset-base")
        tokenizer.save_pretrained(output_path /  "qry_encoder_dpr")

        tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
        tokenizer.save_pretrained(output_path /  "ctx_encoder_dpr")


    def training_step(self, batch, batch_idx):
        loss, predictions, labels = self.forward(**batch)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return {"loss": loss, "preds": predictions, "labels": labels}

    def training_step_end(self, outputs):
        preds = outputs["preds"]
        labels = outputs["labels"]
        for name, metric in self.metrics["train"].items():
            self.log(f"train_{name}", metric(preds, labels), prog_bar=True)

        return outputs['loss'].sum()


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, predictions, labels = self.forward(**batch)    
        self.log("val_loss", loss, on_step=True, on_epoch=False)

        return {"loss": loss, "preds": predictions, "labels": labels}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.log("val_loss", loss, prog_bar=True)
        for name, metric in self.metrics["val"].items():
            self.log(f"val_{name}", metric(preds, labels), prog_bar=True)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        loss, predictions, labels = self.forward(**batch)    
        self.log("test_loss", loss, on_step=True, on_epoch=False)

        return {"loss": loss, "preds": predictions, "labels": labels}


    def test_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        self.log("test_loss", loss, prog_bar=True)
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
