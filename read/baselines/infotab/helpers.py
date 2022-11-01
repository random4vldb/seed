import re 
import datetime
import pandas as pd
import torch
import collections
import inflect
from loguru import logger
import json

def is_date(string):
    match = re.search("\d{4}-\d{2}-\d{2}", string)
    if match:
        try:
            date = datetime.datetime.strptime(match.group(), "%Y-%m-%d").date()
        except:
            return False
        return True
    else:
        return False


def load_sentences(file, skip_first=True, single_sentence=False):
    """Loads sentences into process-friendly format for a given file path.
    Inputs
    -------------------
    file    - str or pathlib.Path. The file path which needs to be processed
    skip_first      - bool. If True, skips the first line.
    single_sentence - bool. If True, Only the hypothesis statement is chosen.
                            Else, both the premise and hypothesis statements are
                            considered. This is useful for hypothesis bias experiments.

    Outputs
    --------------------
    rows    - List[dict]. Consists of all data samples. Each data sample is a
                    dictionary containing- uid, hypothesis, premise (except hypothesis
                    bias experiment), and the NLI label for the pair
    """
    rows = []
    df = pd.read_csv(file, sep="\t")
    for idx, row in df.iterrows():
        # Takes the relevant elements of the row necessary. Putting them in a dict,
        if single_sentence:
            sample = {
                "uid": row["annotator_id"],
                "hypothesis": row["hypothesis"],
                "label": int(row["label"]),
            }
        else:
            sample = {
                "uid": row["index"],
                "hypothesis": row["hypothesis"],
                "premise": row["premise"],
                "label": int(row["label"]),
            }

        rows.append(sample)  # Append the loaded sample
    return rows


def infotab_tokenize(tokenizer, pt_dict, single_sentence=False):
    if single_sentence:
        encoded_inps = tokenizer(
            pt_dict["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=504,
            return_tensors="pt",
        )
    else:
        encoded_inps = tokenizer(
            pt_dict["premise"],
            pt_dict["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=504,
            return_tensors="pt",
        )

    # Some models do not return token_type_ids and hence
    # we just return a list of zeros for them. This is just
    # required for completeness.
    if "token_type_ids" not in encoded_inps.keys():
        encoded_inps["token_type_ids"] = torch.zeros_like(
            encoded_inps["input_ids"]
        )
    return encoded_inps


def infotab_linearize(index, row):
    if isinstance(row["table"], str):
        row["table"] = json.loads(row["table"])
    table = pd.DataFrame(row["table"]).astype(str)
    obj = collections.defaultdict(list)
    if row["highlighted_cells"]:
        for i, j in row["highlighted_cells"]:
            obj[f"{table.columns[j]}"].append(table.iloc[i, j])
    else:
        for i in range(table.shape[0]):
            for j in range(table.shape[1]):
                obj[f"{table.columns[j]}"].append(table.iloc[i, j])

    if not obj:
        return None
    obj = {x: y if isinstance(y, list) else [y] for x, y in obj.items()}
    
    try:
        title = row["title"]
    except KeyError as e:
        logger.error(f"KeyError: {e}")
        exit()

    para = ""

    if "index" in obj:
        obj.pop("index")

    for key in obj:
        line = ""
        values = obj[key]

        if not key.strip():
            key = "value"
        key = key.replace(" of ", " ")
        key = key.replace(
            " A ", " AB "
        )  # This is a hack to for inflect to work properly
        try:
            key = inflect.plural_noun(key)
        except:
            key = key
        if (len(values) > 1):
            verb_use = "are"
            if is_date("".join(values)):
                para += title + " was " + str(key) + " on "
                line += title + " was " + str(key) + " on "
            else:
                try:
                    para += (
                        "The "
                        + str(key)
                        + " of "
                        + title
                        + " "
                        + verb_use
                        + " "
                    )
                    line += (
                        "The "
                        + str(key)
                        + " of "
                        + title
                        + " "
                        + verb_use
                        + " "
                    )
                except TypeError as e:
                    logger.error(
                        "Error in key: %s in article of title %s",
                        key,
                        title,
                    )
                    exit()
            for value in values[:-1]:
                para += value + ", "
                line += value + ", "
            if len(values) > 1:
                para += "and " + values[-1] + ". "
                line += "and " + values[-1] + ". "
            else:
                para += values[-1] + ". "
                line += values[-1] + ". "
        else:
            verb_use = "is"
            if is_date(values[0]):
                para += (
                    title + " was " + str(key) + " on " + values[0] + ". "
                )
                line += (
                    title + " was " + str(key) + " on " + values[0] + ". "
                )
            else:
                para += (
                    "The "
                    + str(key)
                    + " of "
                    + title
                    + " "
                    + verb_use
                    + " "
                    + values[0]
                    + ". "
                )
                line += (
                    "The "
                    + str(key)
                    + " of "
                    + title
                    + " "
                    + verb_use
                    + " "
                    + values[0]
                    + ". "
                )

    label = row["label"]
    obj = {
        "index": index,
        "table_id": row["table_id"],
        "annotator_id": row["annotator_id"],
        "premise": para,
        "hypothesis": row["hypothesis"],
        "label": label,
    } 
    return obj
