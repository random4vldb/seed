import hydra
import pyrootutils
import jsonlines
from pathlib import Path


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "infotab",
    config_name="sent_selection_convert.yaml",
)
def main(cfg):
    examples = []
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    for input_file in Path(cfg.input_dir).glob("*.jsonl"):
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                examples.append(
                    {
                        "sentence": obj["positive"],
                        "label": 1,
                        "id": obj["id"],
                        "table": obj["table"],
                        "title": obj["title"],
                    }
                )
                examples.append(
                    {
                        "sentence": obj["negative"],
                        "label": 0,
                        "id": obj["id"],
                        "table": obj["table"],
                        "title": obj["title"],
                    }
                )

        with jsonlines.open(Path(cfg.output_dir) / input_file.name, "w") as writer:
            writer.write_all(examples)


if __name__ == "__main__":
    main()
