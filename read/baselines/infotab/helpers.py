import re 
import datetime
import pandas as pd

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