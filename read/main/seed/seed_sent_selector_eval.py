import json
from os import device_encoding

import hydra
import jsonlines
import pandas as pd
import pyrootutils
from loguru import logger
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import BinaryClassificationEvaluator

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.table import linearize_tapex


@hydra.main(version_base="1.2", config_path=root / "config" / "seed", config_name="seed_sent_eval.yaml")
def main(cfg):
    train_sentences1 = []
    train_sentences2 = []
    train_labels = []

    dev_sentences1 = []
    dev_sentences2 = []
    dev_labels = []

    logger.info("Reading data")

    for item in jsonlines.open(cfg.train_path):        
        train_sentences1.extend([item["query"], item["query"]])
        train_sentences2.extend([item["positive"], item["negative"]])
        train_labels.extend([1, 0])

    for item in jsonlines.open(cfg.dev_path):        
        dev_sentences1.extend([item["query"], item["query"]])
        dev_sentences2.extend([item["positive"], item["negative"]])
        dev_labels.extend([1, 0])
    

    logger.info("Loading model")
    model = SentenceTransformer(cfg.model_path)

    logger.info("Evaluating model")
    evaluator = BinaryClassificationEvaluator(train_sentences1, train_sentences2, train_labels, name="temp/seed/sent_selection/result", show_progress_bar=True)

    logger.info(f"Evaluation result: {evaluator(model)}")


if __name__ == "__main__":
    main()

