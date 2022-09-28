from .module import SentSelectModule
from transformers import AutoTokenizer
from pytorch_lightning import Trainer

class SentenceSelector:
    def __init__(self, model_name_or_path: str, tokenizer: str, cfg) -> None:
        self.model = SentSelectModule.load_from_checkpoint(model_name_or_path).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.trainer = Trainer(devices=cfg.devices, accelerator="gpu")


    def __call__(self, queries, sentences, indices):
        all_preds = []
        for i in range(0, len(queries), 2):
            inputs = self.tokenizer(queries[i: i + 2], sentences[i: i + 2], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = self.trainer.predict(self.model, inputs)

            preds = outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
        return all_preds

