import hydra
import pyrootutils
import jsonlines
from loguru import logger
from torchmetrics import Accuracy, F1Score, Precision, Recall

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


logger.add("logs/debug.log", rotation="1 week")

@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "pipeline",
    config_name="pipeline_generate.yaml",
)
def main(cfg):
    with jsonlines.open(cfg.prediction_file) as reader:
        predictions = list(reader)
        id2result = {}

        for pred in predictions:
            if pred["id"] not in id2result:
                id2result[pred["id"]] = pred
            elif id2result[pred["id"]]["label"] == 1:
                continue
            else:
                if pred["label"] == 1:
                    id2result[pred["id"]] = pred
        
    name2metrics = {
        "accuracy": Accuracy(),
        "f1": F1Score(num_classes=2),
        "precision": Precision(num_classes=2),
        "recall": Recall(num_classes=2),
    }
    
    with jsonlines.open(cfg.input_file) as reader:
        data = list(reader)
        for example in data:
            id = example["id"]
            label = example["label"]
            prediction = id2result[id]["label"]
            for name, metric in name2metrics.items():
                metric.update(prediction, label)

    for name, metric in name2metrics.items():
        logger.info(f"{name}: {metric.compute()}")
        metric.reset()



if __name__ == "__main__":
    main()
