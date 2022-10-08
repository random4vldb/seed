from turtle import update
from tango import Step, JsonFormat, Format
from tango.common import DatasetDict
from typing import Dict
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tango.common.tqdm import Tqdm
from torch.utils.data import DataLoader
import evaluate


@Step.register("eval::classification")
class ClassificationScoreStep(Step[Dict[str, Tensor]]):
    FORMAT: Format = JsonFormat()

    def run(  # type: ignore
        self,
        model,
        dataset_dict,
        test_split,
        batch_size,
    ) -> Dict[str, Tensor]:

        name2metrics = {
            "accuracy": evaluate.load("accuracy"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1"),
        }

        dataloader = DataLoader(dataset_dict[test_split], batch_size=batch_size, shuffle=False)

        for batch in Tqdm.tqdm(dataloader, desc="Calculating scores"):
            y_hat = model(**batch)
            preds = y_hat.logits.argmax(dim=1)
            for metric in name2metrics.values():
                metric.add_batch(predictions=preds, references=batch["labels"])

        return {name: metric.compute() for name, metric in name2metrics.items()}
