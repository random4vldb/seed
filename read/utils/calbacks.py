from typing import Any, Dict, Optional

import evaluate
from tango.integrations.torch import EvalCallback, TrainCallback

@EvalCallback.register("eval::classification_score_callback")
class ClassificationEvalCallback(EvalCallback):
    VERSION: Optional[str] = "003"

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


@TrainCallback.register("train::classification_score_callback")
class ClassificationTrainCallback(TrainCallback):
    VERSION: Optional[str] = "003"

    name2metrics = {
        "accuracy": evaluate.load("accuracy"),
        "f1": evaluate.load("f1"),
        "precision": evaluate.load("precision"),
        "recall": evaluate.load("recall"),
    }

    def post_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch_outputs: Dict[str, Any]
    ):
        preds = val_batch_outputs["predictions"]
        labels = val_batch_outputs["labels"]
        for name, metric in self.name2metrics.items():
            metric.add_batch(predictions=preds, references=labels)

    def post_eval_loop(
        self, step: int, epoch: int, val_metric: float, best_val_metric: float
    ) -> None:
        for name, metric in self.name2metrics.items():
            self.logger.info(f"{name}: {metric.compute()}")
