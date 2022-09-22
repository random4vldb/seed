import hydra
import pyrootutils
import json
import pandas as pd
import random
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

@hydra.main(version_base="1.2", config_path=root / "config" / "seed", config_name="seed_cell_generate.yaml")
def main(cfg):
    examples = []

    for input_file in Path(cfg.input_dir).glob("*.jsonl"):
        
        df = pd.read_json(input_file, lines=True)
        df["table"] = df["table"].apply(lambda x: pd.DataFrame(json.loads(x)))

        for i, row in df.iterrows():
            for cell in row["highlighted_cells"]:
                print(row["table"])
                examples.append({
                    "table": row["table"].iloc[cell[0], :].values.tolist(),
                    "cell": cell,
                    "value": row["table"].iloc[cell[0], cell[1]],
                    "column": row["table"].columns[cell[1]],
                    "label": 1,
                    "title": row["table_page_title"],
                    "sentence": row["subtable_metadata_str"],
                })

            other_indices = [(i, j) for i in range(len(row["table"])) for j in range(len(row["table"].columns)) if (i, j) not in row["highlighted_cells"]]
            random_indices = random.sample(other_indices, len(row["highlighted_cells"]))

            for cell in random_indices:
                examples.append({
                    "context": row["table"].iloc[cell[0], :].values.tolist(),
                    "cell": cell,
                    "value": row["table"].iloc[cell[0], cell[1]],
                    "column": row["table"].columns[cell[1]],
                    "label": 0,
                    "title": row["table_page_title"],
                    "sentence": row["subtable_metadata_str"],
                })
        

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        with jsonlines.open(Path(cfg.output_dir) / input_file.name, "w") as f:
            f.write_all(examples)

            

if __name__ == "__main__":
    main()
