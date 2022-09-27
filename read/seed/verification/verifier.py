from read.seed.verification.module import Seed3Module
from transformers import AutoTokenizer
import torch


class TableVerifier:
    def __init__(self, model_path, tokenizer):
        self.nli_model = Seed3Module.load_from_checkpoint(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        

    def __call__(self, linearized_tables, sentences):
        encodings = self.tokenizer(linearized_tables, sentences, truncation=True, padding=True, return_tensors="pt")
        outputs = self.nli_model(**encodings).logits

        return torch.argmax(outputs, dim=1) == 1, torch.softmax(outputs, dim=1)
