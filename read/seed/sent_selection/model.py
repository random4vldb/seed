from sentence_transformers import SentenceTransformer, util
from .module import SentSelectModule
from transformers import AutoTokenizer

class SentenceSelector:
    def __init__(self, model_name_or_path: str, tokenizer: str) -> None:
        self.model = SentSelectModule.load_from_checkpoint(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    def __call__(self, queries, sentences, indices):
        inputs  = self.tokenizer(queries, sentences, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)

        preds = outputs.logits.argmax(dim=1)
        return preds

