import jsonlines
import pandas as pd
import collections
import numpy as np
import wikitextparser as wtp
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

@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "preprocess",
    config_name="wiki_history.yaml",
)
def main(cfg):
    tables = jsonlines.open(cfg.input_file)

    key_to_tables = collections.defaultdict(list)
    logger.info("Loading tables")
    for table_obj in tables:
        table = pd.DataFrame(table_obj["table"]).astype(str).applymap(lambda x: wtp.remove_markup(x))
        for col in table.columns:
            table = table.rename(columns={col: wtp.remove_markup(col)})
        key = (
            "|".join(table.columns.values),
            table_obj["page_id"],
            table_obj["page_title"],
        )
        
        try:
            id = int(table_obj["revision_id"])
        except:
            id = 0
        
        key_to_tables[key].append((id, table))

    data = []

    logger.info("Processing tables")

    for key, tables in key_to_tables.items():
        tables.sort(key=lambda x: x[0])
        true_table = tables[-1][1]
        existing_rows = set()
        for revision_id, table in tables[:-1]:
            print("Table", table)
            print("True table", true_table)
            try:
                ground_truth = (table == true_table)
            except ValueError:
                continue
            rows = np.where(np.any(~ground_truth, axis=1))[0].tolist()
            table["page_title"] = [key[2]] * len(table)

            for row in rows:
                example = {
                    "table": table.iloc[[row], :].to_json(orient="records"),
                    "label": False,
                    "true_table": true_table.iloc[[row], :].to_json(orient="records"),
                    "revision_id": revision_id,
                    "title": key[2],
                }

                data.append(example)

                if row not in existing_rows:
                    existing_rows.add(row)
                    example = {
                        "table": true_table.iloc[[row], :].to_json(orient="records"),
                        "label": False,
                        "true_table": true_table.iloc[[row], :].to_json(orient="records"),
                        "revision_id": revision_id,
                        "title": key[2],
                    }
                    data.append(example)

    logger.info("Writing data")
    with jsonlines.open(cfg.output_file, mode="w") as writer:
        writer.write_all(data)


if __name__ == "__main__":
    main()

            
                
