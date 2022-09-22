from sentence_transformers import SentenceTransformer, InputExample, models, losses
from sentence_transformers.losses import TripletDistanceMetric
from sentence_transformers.evaluation import BinaryClassificationEvaluator
import jsonlines
import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
import pyrootutils
from loguru import logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.table import linearize_tapex


@hydra.main(version_base="1.2", config_path=root / "config" / "seed", config_name="seed_sent_train.yaml")
def main(cfg):
    examples = []

    train_sentences1 = []
    train_sentences2 = []
    train_labels = []

    dev_sentences1 = []
    dev_sentences2 = []
    dev_labels = []

    logger.info("Reading data")

    for item in jsonlines.open(cfg.train_path):
        examples.append(InputExample(texts=[item["query"], item["positive"], item["negative"]]))

        train_sentences1.extend([item["query"], item["query"]])
        train_sentences2.extend([item["positive"], item["negative"]])
        train_labels.extend([1, 0])

    for item in jsonlines.open(cfg.dev_path):        
        dev_sentences1.extend([item["query"], item["query"]])
        dev_sentences2.extend([item["positive"], item["negative"]])
        dev_labels.extend([1, 0])

    if cfg.get("train"):
        word_embedding_model = models.Transformer('facebook/bart-base', max_seq_length=512)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

        train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
        train_loss = losses.TripletLoss(model=model, triplet_margin=0.5, distance_metric=TripletDistanceMetric.COSINE)

        model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=5, show_progress_bar=True, output_path=cfg.output_dir)

    if cfg.get("dev"):
        model = SentenceTransformer(cfg.output_dir)
        evaluator = BinaryClassificationEvaluator(dev_sentences1, dev_sentences2, dev_labels, name="temp/seed/sent_selection/result", show_progress_bar=True)
        logger.info(f"Evaluation result: {evaluator(model)}")

if __name__ == "__main__":
    main()

