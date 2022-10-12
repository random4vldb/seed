from tango import Step
from pathlib import Path
import jsonlines
import pandas as pd
import json


def sub_table(table, highlighted_cells):
    table = pd.DataFrame(json.loads(table))
    cells = zip(*highlighted_cells)
    cells = [list(x) for x in cells]
    sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)
    return sub_table


@Step.register("seed_correction_generate")
class SeedCorrectionGenerate(Step):
    def run(self, input_path):
        for file in Path(input_path).glob("*.jsonl"):
            with jsonlines.open(file, "r") as reader:
                for obj in reader:
                    table = sub_table(obj["table"], obj["highlighted_cells"])
                    if len(table) != 1:
                        continue
                    for col in table.columns:
                        yield {
                            "table": sub_table(obj["table"], obj["highlighted_cells"]),
                            "question": col,
                            "context": obj["sentence"],
                            "answer": table[col][0],
                        }
