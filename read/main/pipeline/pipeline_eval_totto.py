import hydra
import pyrootutils
import jsonlines
from loguru import logger
import random

from read.pipeline.seed import SEEDPipeline

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.metrics.pipeline import PipelineEvaluator

logger.add("logs/debug.log", rotation="1 week")

@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "pipeline",
    config_name="pipeline_totto.yaml",
)
def main(cfg):
    with jsonlines.open(cfg.input_file) as reader:
        data = list(reader)
        random.seed(21)
        random.shuffle(data)
        data = data
        data = data[:100]

    logger.info(f"Loaded {len(data)} examples with {sum([x['label'] for x in data])} positives", )

    batch = []    
    predictions = []
    golds = []
    gold_batch = []
    report = PipelineEvaluator()
    pipeline = SEEDPipeline.init_with_config(cfg, report)


    for i in range(len(data)):
        data[i]["linearized_table"] = data[i]["subtable_metadata_str"]
        data[i]["title"] = data[i]["table_page_title"]

        batch.append(data[i])
        gold_batch.append(data[i]["label"])

    logger.info(f"Processing batch of size {len(batch)}")
    result = pipeline(batch)
    logger.info(f"Got {len(result)} predictions")
    predictions.extend(result)
    golds.extend(gold_batch)
    batch = []
    gold_batch = []


    if len(batch) > 0:
        result = pipeline(batch)
        predictions.extend(result)
        golds.extend(gold_batch)


    report.report()

if __name__ == "__main__":
    main()