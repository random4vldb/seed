from dataclasses import dataclass
from .module import SentSelectModule
from transformers import AutoTokenizer
from pytorch_lightning import Trainer
from datasets import Dataset
from torch.utils.data import DataLoader

class SentenceSelector:
    def __init__(self, model_name_or_path: str, tokenizer: str, cfg) -> None:
        self.model = SentSelectModule.load_from_checkpoint(model_name_or_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.trainer = Trainer(devices=cfg.devices, accelerator="gpu")


    def __call__(self, queries, sentences, indices):
        all_preds = []
        inputs = self.tokenizer(list(zip(queries, sentences)), return_tensors="pt", padding=True, truncation=True)

        dataset = Dataset.from_dict(inputs)

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

        outputs = self.trainer.predict(self.model, dataloader, return_predictions=True)

        preds = outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        return all_preds

