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
    DOCUMENT_RETRIEVAL = "document_retrieval"
    SENTENCE_SELECTION = "sentence_selection"
    TABLE_VERIFICATION = "table_verification"

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
        print(self.stage2metrics)

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
            