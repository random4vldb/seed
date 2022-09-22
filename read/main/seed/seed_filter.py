import collections
import enum
import json
import random
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

import hydra
import jsonlines
import numpy as np
import pandas as pd
import pyrootutils
import spacy
from loguru import logger
import tqdm
import re

nlp = spacy.load("en_core_web_sm")

logger.add("logs/seed_filter.log")

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

def empty_ratio(table):
    return table.replace("", np.nan).isna().mean().mean()


def numeric_ratio(table):
    values = table.values.flatten().tolist()
    numeric = 0
    total = 0
    for value in values:
        total += 1
        value = re.search('[A-Za-z]', value)
        if not value:
            numeric += 1

    if total == 0:
        return 1.0
    return numeric * 1.0 / total


def filter_sentence(jobj):
    doc = nlp(jobj["sentence_annotations"][0]["original_sentence"])
    for token in doc:
        if token.pos_ == "VERB":
              return True
    return False


def filter_tables(df):
    df["num_cells"] = df["highlighted_cells"].apply(lambda x: len(x))
    df["num_highlighted_rows"] = df["highlighted_cells"].apply(lambda x: len(set([c[0] for c in x])))
    df["num_highlighted_cols"] = df["highlighted_cells"].apply(lambda x: len(set([c[1] for c in x])))
    df["num_rows"] = df["table"].apply(lambda x: len(x))
    df["num_cols"] = df["table"].apply(lambda x: len(x.columns))
    df["num_numeric_headers"] = df["table"].apply(lambda x: len([c for c in x.columns if c.isdigit()]))

    tables = df["table"].tolist()

    pool = Pool()
    df["empty_rate"] = pool.map(empty_ratio, tables)
    pool.close()


    pool = Pool()
    df["numeric_rate"] = pool.map(numeric_ratio, tables)
    pool.close()

    df = df[df["numeric_rate"] + df["empty_rate"] < 0.33]    
    df = df[df["num_cells"] > 1]
    df = df[df["num_highlighted_cols"] <= 10]
    df = df[df["num_highlighted_rows"] <= 3]
    df = df[df["num_numeric_headers"] != df["num_cols"]]
    df = df[~df["table_page_title"].str.contains("List")]

    df = df.drop(["num_cells", "num_rows", "num_cols", "num_highlighted_rows", "num_highlighted_cols", "empty_rate", "numeric_rate"], axis=1)
    df["sentence"] = df["sentence_annotations"].apply(lambda x: x[0]["original_sentence"])

    pool = Pool()

    jobjs = df.to_dict("records")
    has_verb = pool.map(filter_sentence, jobjs)
    df["has_verb"] = has_verb
    df = df[df["has_verb"]]

    pool.close()

    return df


def valid_columns(table, highlighted_cells):
    empty_count_by_col = table.replace("", np.nan).isna().sum().to_dict()
    highlighted_cols = [x[1] for x in highlighted_cells]
    
    valid_columns = []
    for idx, col in enumerate(table.columns):
        if empty_count_by_col[col] > 0 or idx not in highlighted_cols:
            continue
        valid_columns.append(idx)
        
    return valid_columns


def generate_negatives(jobj):
    idx, jobj = jobj
    table, highlighted_cells, valid_columns = jobj["table"], jobj["highlighted_cells"], jobj["valid_columns"]
    replacing_value = None
    while valid_columns:
        valid_column = random.choice(valid_columns)
        replaced_rows = [i for i in range(len(table)) if [i, valid_column] in highlighted_cells]
        if not replaced_rows:
            valid_columns.remove(valid_column)
            continue
        
        replaced_value = "!!!REPLACED!!!"
        while replaced_value not in jobj["subtable_metadata_str"]:
            replaced_row = random.choice(replaced_rows)
            replaced_value = table.iloc[replaced_row, valid_column]
            replaced_rows.remove(replaced_row)
        if replaced_value == "!!!REPLACED!!!":
            continue
        replacing_rows = [i for i in range(len(table)) if i != replaced_row]
        if not replacing_rows:
            valid_columns.remove(valid_column)
            continue
            
        while replacing_rows:
            replacing_row = random.choice(replacing_rows)
            replacing_value = table.iloc[replacing_row, valid_column]
            if replacing_value != replaced_value:
                sent = jobj["subtable_metadata_str"].replace(replaced_value, replacing_value, 1)
                return idx, sent, (replacing_value, replaced_value, replaced_row, valid_column)
            replacing_rows.remove(replacing_row)
        valid_columns.remove(valid_column)

    return idx, None, None

def generate(df):
    df["valid_columns"] = df.apply(lambda x: valid_columns(x["table"], x["highlighted_cells"]), axis=1)

    df["num_rows"] = df["table"].apply(lambda x: len(x))

    df = df[df.valid_columns.astype(bool)]
    valid_df = df[df["num_rows"] >= 2].copy(deep=True)

    jobjs = valid_df.to_dict(orient="records")

    pool = Pool()
    
    negatives = list(tqdm.tqdm(pool.imap_unordered(generate_negatives, enumerate(jobjs)), total=len(jobjs)))
    pool.close()

    logger.info("Length of negatives with nulls: {}".format(len(negatives)))

    ordered_negatives = sorted(negatives, key=lambda x: x[0])
    ordered_negatives = [(x[1], x[2]) for x in ordered_negatives]


    valid_df["negatives"] = [x[1] for x in ordered_negatives]
    valid_df["subtable_metadata_str"] = [x[0] for x in ordered_negatives]
    valid_df = valid_df[valid_df["negatives"].notnull()]


    df.loc[: ,"negatives"] = None

    final_df = pd.concat([df, valid_df], axis=0)

    return final_df

@hydra.main(
    version_base="1.2", config_path=root / "config" / "seed", config_name="seed_filter.yaml"
)
def main(cfg):
    data_dir = Path(cfg.data_dir)

    random.seed(21)

    for data_file in data_dir.iterdir():
        if data_file.is_dir():
            continue
        logger.info(f"Working on {data_file}")

        df = pd.read_json(data_file, lines=True)
        df["table"] = df["table"].apply(lambda x: pd.DataFrame(json.loads(x)))

        if "sentence_annotations" not in df.columns:
            continue

        df = filter_tables(df)
        saved_df = df.copy()
        saved_df["table"] = saved_df["table"].apply(lambda x: x.to_json(orient="records"))

        logger.info("Generate negatives with len(df) = {}".format(len(df)))

        Path(cfg.filtered_dir).mkdir(exist_ok=True, parents=True)

        saved_df.to_json(Path(cfg.filtered_dir) / data_file.name, orient="records", lines=True)

        final_df = generate(df)

        logger.info("Final df with len(df) = {} and {} negatives".format(len(final_df), len(final_df) - len(df)))

        Path(cfg.augmented_dir).mkdir(exist_ok=True, parents=True)

        final_df["label"] = final_df["negatives"].apply(lambda x: 1 if x is None else 0)
        final_df["negatives"] = final_df["negatives"].apply(lambda x: "" if x is None else json.dumps(x))
        final_df.to_json(Path(cfg.augmented_dir) / data_file.name, lines=True, orient="records")

        jobjs = final_df.to_dict("records")
        

        id2examples = collections.defaultdict(list)
        for obj in jobjs:
            id2examples[obj["example_id"]].append(obj)

        id2triplet = dict(filter(lambda x: len(x[1]) == 2, id2examples.items()))

        triplets = []

        logger.info("Generating triplets")

        for key, (ex1, ex2) in id2triplet.items():
            triple = {}
            assert ex1["negatives"] == "" or ex2["negatives"] == ""
            assert ex1["negatives"] == "" or ex2["negatives"] == ""
            
            if ex1["negatives"] == "":
                triple["positive"] = ex2["negatives"]
                triple["label"] = 1

            if ex1["negatives"] == "":
                triple["positive"] = ex1["subtable_metadata_str"]
                triple["negative"] = ex2["subtable_metadata_str"]
                triple["positive_table"] = ex1["table"].to_json(orient="records")
                triple["negative_table"] = ex2["table"].to_json(orient="records")
                triple["note"] = ex2["negatives"]
            else:
                triple["negative"] = ex1["subtable_metadata_str"]
                triple["positive"] = ex2["subtable_metadata_str"]
                triple["positive_table"] = ex2["table"].to_json(orient="records")
                triple["negative_table"] = ex1["table"].to_json(orient="records")
                triple["note"] = ex1["negatives"]
                
            triple["sentence"] = ex1["sentence"]
            triple["title"] = ex1["table_page_title"]
            triplets.append(triple)

        Path(cfg.triplet_dir).mkdir(exist_ok=True, parents=True)

        with jsonlines.open(Path(cfg.triplet_dir) / data_file.name, "w") as writer:
            writer.write_all(triplets)


if __name__ == "__main__":
    main()
