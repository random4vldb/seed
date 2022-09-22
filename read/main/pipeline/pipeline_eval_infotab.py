import hydra
import pyrootutils
import jsonlines
import torchmetrics
from loguru import logger
import torch
import random
import json

from read.pipeline.seed import SEEDPipeline

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.exp import ResultAnalyzer
from read.utils.table import infotab2totto

logger.add("logs/debug.log", rotation="1 week")

@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "pipeline",
    config_name="pipeline_infotab.yaml",
)
def main(cfg):
    with jsonlines.open(cfg.input_file) as reader:
        linearized_tables = []
        labels = []
        data = list(reader)
        random.seed(42)
        random.shuffle(data)
        data = data
        for raw_obj in data:
            obj = infotab2totto(raw_obj)
            raw_obj["subtable_metadata_str"] = obj
            linearized_tables.append(obj)
            labels.append(raw_obj["label"] == 2)

    logger.info(f"Loaded {len(labels)} examples with {sum(labels)} positives")

    f1 = torchmetrics.F1Score(num_classes=2)
    precision = torchmetrics.Precision(num_classes=2)
    recall = torchmetrics.Recall(num_classes=2)
    acc = torchmetrics.Accuracy(num_classes=2)

    batch = []    
    predictions = []
    golds = []
    gold_batch = []
    report = ResultAnalyzer(data)
    pipeline = SEEDPipeline(cfg, report)


    for i in range(len(linearized_tables)):
        batch.append(linearized_tables[i])
        gold_batch.append(labels[i])

        if len(batch) == cfg.batch_size:
            result = pipeline(batch)
            predictions.extend(result)
            golds.extend(labels[i - cfg.batch_size + 1 : i + 1])
            batch = []

            logger.info(f"Processed {i} examples")
            for metric in [f1, precision, recall, acc]:
                logger.info(f"{metric.__class__.__name__}: {metric(torch.tensor(predictions), torch.tensor(golds))}")

    if len(batch) > 0:
        result = pipeline(batch)
        predictions.extend(result)
        golds.extend(labels[-len(batch) :])
    for metric in [f1, precision, recall, acc]:
        logger.info(f"{metric.__class__.__name__}: {metric(torch.tensor(predictions), torch.tensor(golds))}")

    report.print("debug.txt")

if __name__ == "__main__":
    main()