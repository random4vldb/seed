from tango.integrations.torch import TrainCallback, TrainConfig, TrainingEngine
from typing import Any, Dict
from tango.workspace import Workspace
from tango.common.dataset_dict import DatasetDictBase
from torch.utils.data import DataLoader
import evaluate
from typing import Optional


@TrainCallback.register("classify_score_callback")
class ClassificationScoresCallback(TrainCallback):
    def __init__(
        self,
        workspace: Workspace,
        train_config: TrainConfig,
        training_engine: TrainingEngine,
        dataset_dict: DatasetDictBase,
        train_dataloader: DataLoader,
        validation_dataloader: Optional[DataLoader] = None,
    ) -> None:
        super().__init__(
            workspace,
            train_config,
            training_engine,
            dataset_dict,
            train_dataloader,
            validation_dataloader,
        )
        
        self.name2metrics = {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }

    def post_val_batch(
        self, step: int, val_step: int, epoch: int, val_batch_outputs: Dict[str, Any]
    ) -> None:
        preds = val_batch_outputs["predictions"]
        labels = val_batch_outputs["labels"]
        for name, metric in self.name2metrics.items():
            metric.add_batch(predictions=preds, references=labels)


    def post_val_loop(self, step: int, epoch: int, val_metric: float, best_val_metric: float) -> None:
        name2result = {}
        for name, metric in self.name2metrics.items():
            name2result[name] = metric.compute()
            metric.reset()
        self.workspace.log_metrics(name2result, step=step, epoch=epoch)
