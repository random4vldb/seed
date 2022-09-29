import hydra
import pyrootutils
import jsonlines

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.pipeline.baselines.tapas import TapasPipeline

@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "pipeline",
    config_name="pipeline_totto.yaml",
)
def main(cfg):
    baseline = TapasPipline(cfg)
    
    examples = jsonlines.open(cfg.input_file, "r")

    preds = baseline.predict(examples)

    jsonlines.open(cfg.output_file, "w").write_all(preds)
