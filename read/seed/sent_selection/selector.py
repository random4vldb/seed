from .module import SentSelectModule
from transformers import AutoTokenizer

class SentenceSelector:
    def __init__(self, model_name_or_path: str, tokenizer: str) -> None:
        self.model = SentSelectModule.load_from_checkpoint(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    def __call__(self, queries, sentences, indices):
        all_preds = []
        for i in range(0, len(queries), 8):
            inputs = self.tokenizer(queries[i: i + 8], sentences[i: i + 8], return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)

            preds = outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()
            all_preds.extend(preds)
        return all_preds

