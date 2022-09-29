from collections import defaultdict
from ..seed import SEEDPipeline
from transformers import TapasForSequenceClassification, TapasTokenizer

class TapasPipeline(SEEDPipeline):
    def __init__(self, cfg):
        self.sent_selector = TapasForSequenceClassification.from_pretrained(cfg.sent_selector.model_name_or_path)
        self.sent_tokenizer = TapasTokenizer.from_pretrained(cfg.sent_selector.tokenizer)
        self.verifier = TapasForSequenceClassification.from_pretrained(cfg.verifier.model_name_or_path)
        self.verifier_tokenizer = TapasTokenizer.from_pretrained(cfg.verifier.tokenizer)


    def predict(self, examples):
        inputs = self.sent_tokenizer(examples["table"], examples["sentence"], return_tensors="pt", padding=True, truncation=True)

        outputs = self.sent_selector(**inputs)

        selected_examples = defaultdict(list)
        sent2verification = {}
        for i, pred in enumerate(outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()):
            if pred == 1:
                sent2verification[i] = len(sent2verification)
                selected_examples["table"].append(examples["table"][i])
                selected_examples["sentence"].append(examples["sentence"][i])
        
        result = [False] * len(examples)
        if len(selected_examples) != 0:
            inputs = self.verifier_tokenizer(selected_examples["table"], selected_examples["sentence"], return_tensors="pt", padding=True, truncation=True)
            outputs = self.verifier(**inputs)
            for sent_idx, verification_idx in sent2verification.items():
                result[sent_idx] = outputs.logits[verification_idx].argmax(dim=0).detach().cpu().numpy().tolist() == 1

        return [{"id": examples[i]["id"], "label": result[i]} for i in range(len(examples))]
