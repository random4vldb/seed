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
    config_name="verification_convert.yaml",
)
def main(cfg):
    examples = []
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    for input_file in Path(cfg.input_dir).glob("*.jsonl"):
        with jsonlines.open(input_file) as reader:
            for obj in reader:
                print(obj)
                examples.append(
                    {
                        "sentence": obj["sentence"],
                        "label": 1,
                        "table": obj["positive_table"],
                        "title": obj["title"],
                        "highlighted_cells": obj["highlighted_cells"],
                    }
                )
                examples.append(
                    {
                        "sentence": obj["sentence"],
                        "label": 0,
                        "table": obj["negative_table"],
                        "title": obj["title"],
                        "highlighted_cells": obj["highlighted_cells"],
                    }
                )

        with jsonlines.open(Path(cfg.output_dir) / input_file.name, "w") as writer:
            writer.write_all(examples)


if __name__ == "__main__":
    main()
