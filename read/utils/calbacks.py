from typing import Any, Dict

import evaluate
from tango.common.dataset_dict import DatasetDictBase
from tango.integrations.torch import EvalCallback, Model
from tango.workspace import Workspace
from torch.utils.data import DataLoader



@EvalCallback.register("classify_score_callback")
class ClassificationEvalCallback(EvalCallback):
    name2metrics = {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }

    def post_batch(self, step: int, batch_outputs: Dict[str, Any]) -> None:
        preds = batch_outputs["predictions"]
        labels = batch_outputs["labels"]
        for name, metric in self.name2metrics.items():
            metric.add_batch(predictions=preds, references=labels)

    def post_eval_loop(self, aggregated_metrics: Dict[str, float]) -> None:
        for name, metric in self.name2metrics.items():
            aggregated_metrics[name] = metric.compute()
