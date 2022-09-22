import collections
from loguru import logger
from torchmetrics import (
    Accuracy,
    F1Score,
    Precision,
    Recall,
    RetrievalHitRate,
    RetrievalPrecision,
    RetrievalRecall,
)
import torch


class PipelineModule:
    DOCUMENT_RETRIEVAL = 1
    SENTENCE_SELECTION = 2
    TABLE_VERIFICATION = 3

class PipelineEvaluator:


    def __init__(self) -> None:
        self.stage2metrics = collections.defaultdict(dict)
        
        for stage in [PipelineModule.DOCUMENT_RETRIEVAL, PipelineModule.SENTENCE_SELECTION, PipelineModule.TABLE_VERIFICATION]:
            if stage == PipelineModule.DOCUMENT_RETRIEVAL:
                for metric in [RetrievalPrecision, RetrievalRecall, RetrievalHitRate]:
                    self.stage2metrics[stage][metric.__class__.__name__] = metric(k=10)
            else:
                for metric in [Accuracy, F1Score, Precision, Recall]:
                    self.stage2metrics[stage][metric.__class__.__name__] = metric()

    def update(self, stage, values, golds, indices):
        for metric in self.stage2metrics[stage].values():
            if stage == PipelineModule.DOCUMENT_RETRIEVAL:
                metric.update(torch.tensor(values), torch.tensor(golds), torch.tensor(indices))
            else:
                metric.update(torch.tensor(values), torch.tensor(golds))

    def report(self):
        for stage, metrics in self.stage2metrics.items():
            for metric, value in metrics.items():
                logger.info(f"{stage} {metric}: {value.compute()}")
            