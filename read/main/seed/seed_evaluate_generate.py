import collections
import json
import random
from pathlib import Path
import copy
import jsonlines
import hydra

import pandas as pd
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


@hydra.main(
    version_base="1.2", config_path=root / "config" / "seed", config_name="seed_evaluate_generate.yaml"
)
def main(args):
    table2category = {
        row["table_id"]: row["category"]
        for _, row in pd.read_csv(args.input_path, sep="\t").iterrows()
    }
    category2tables = collections.defaultdict(list)

    for table, category in table2category.items():
        category2tables[category].append(table)
    type2keys = json.load(open(args.category_path, "r"))
    key2type = {key: category for category, keys in type2keys.items() for key in keys}

    category_type2values = collections.defaultdict(list)
    table2valid_keys = collections.defaultdict(list)


    tables = []

    for table_path in Path(args.tables_path).glob("*.json"):
        table_id = table_path.stem
        table = json.load(open(table_path, "r"))
        tables.append(table)
        if len(table) == 0:
            continue
        if table_id not in table2category:
            continue
        category = table2category[table_id]

        for k, v in table.items():
            if k in key2type:
                category_type2values[f"{category}_{key2type[k]}"].extend(v)
                table2valid_keys[table_id].append(k)

    examples = []

    for table_path in Path(args.tables_path).glob("*.json"):
        table_id = table_path.stem
        table = json.load(open(table_path, "r"))
        new_table = copy.deepcopy(table)
        if table2valid_keys[table_id] == []:
            continue
        random_key = random.choice(table2valid_keys[table_id])
        print(random_key)
        random_value = random.choice(category_type2values[f"{table2category[table_id]}_{key2type[random_key]}"])

        new_table[random_key] = [random_value]
        examples.append({"table": table, "label": True, "counter_fact": "", "title": table["title"][0]})
        examples.append({"table": new_table, "label": False, "counter_fact": (None, None, table[random_key][0], random_value[0]), "title": table["title"][0]})

    with jsonlines.open(args.output_path, mode="w") as writer:
        for example in examples:
            writer.write(example)


if __name__ == "__main__":

    
