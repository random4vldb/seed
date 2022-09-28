from read.seed.verification.module import Seed3Module
from transformers import AutoTokenizer
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader



class TableVerifier:
    def __init__(self, model_path, tokenizer, cfg):
        self.nli_model = Seed3Module.load_from_checkpoint(model_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        self.trainer = Trainer(devices=cfg.devices, accelerator="gpu")
        self.cfg = cfg

        

    def __call__(self, linearized_tables, sentences):
        inputs = self.tokenizer(linearized_tables, sentences, truncation=True, padding=True, return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.trainer.predict(self.model, DataLoader(inputs, batch_size=self.cfg.batch_size))

        return torch.argmax(outputs, dim=1) == 1, torch.softmax(outputs, dim=1)
