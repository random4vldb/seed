from re import I
from torchmetrics import RetrievalPrecision, RetrievalRPrecision, RetrievalRecall, RetrievalHitRate
from read.utils.kilt import to_distinct_doc_ids
import torch

class DocRetrievalEvaluator:
    def __init__(self, ks):
        self.ks = ks
        self.k_metrics = {
            k: [RetrievalHitRate(k=k), RetrievalPrecision(k=k), RetrievalRPrecision(k=k), RetrievalRecall(k=k)]
            for k in ks
        }

    def evaluate(self, predictions, golds):
        preds, labels, indices = self.process(predictions, golds)
        result = {}
        for k, metrics in self.k_metrics.items():
            for metric in metrics:
                result[f"{metric.__class__.__name__}@{k}"] = metric(preds, labels, indices)

        return result

    def process(self, predictions, golds):
        preds = []
        labels = []
        indices = []
        print(len(predictions), len(golds))
        for idx, pred in enumerate(predictions):
            gold = golds[idx]
            gold_doc_ids = to_distinct_doc_ids(gold["positive_pids"])
            for i, passage in enumerate(pred["passages"]):
                doc_id = passage["pid"].split(":")[0]
                preds.append(pred["scores"][i])
                labels.append(1 if doc_id in gold_doc_ids else 0)
                indices.append(idx)
        return torch.tensor(preds), torch.tensor(labels), torch.tensor(indices).long()
        

class PassageRetrievalEvaluator:
    def __init__(self, ks):
        self.ks = ks
        self.k_metrics = {
            k: [RetrievalHitRate(k=k), RetrievalPrecision(k=k), RetrievalRPrecision(k=k), RetrievalRecall(k=k)]
            for k in ks
        }
    
    def evaluate(self, preds, labels):
        result = {}
        for k, metrics in self.k_metrics.items():
            for metric in metrics:
                metric(preds, labels)
                result[f"{metric.__class__.__name__}@{k}"] = metric.get_metric()

        return result

    def process(self, preds, labels):
        pass
