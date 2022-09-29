import hydra
import pyrootutils
import jsonlines
from loguru import logger
import random
from pathlib import Path

from read.pipeline.seed import SEEDPipeline

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
    with jsonlines.open(cfg.input_file) as reader:
        data = list(reader)
        random.seed(21)
        random.shuffle(data)
        data = data

    logger.info(f"Loaded {len(data)} examples with {sum([x['label'] for x in data])} positives", )

    batch = []    
    predictions = []
    pipeline = SEEDPipeline.init_with_config(cfg, None)


    for i in range(len(data)):
        data[i]["linearized_table"] = data[i]["subtable_metadata_str"]
        data[i]["title"] = data[i]["table_page_title"]
        data[i]["id"] = i

        batch.append(data[i])

        if len(batch) == cfg.batch_size:
            result = pipeline.generate_nli_data(batch)
            predictions.extend(result)
            batch = []

            logger.info(f"Processed {i} examples")

    if len(batch) > 0:
        result = pipeline.generate_nli_data(batch)
        predictions.extend(result)

    Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(Path(cfg.output_file), "w") as writer:
        for prediction in predictions:
            writer.write(prediction)


if __name__ == "__main__":
    main()