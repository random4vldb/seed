from functools import partial
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyrootutils
import ujson as json
from loguru import logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import read_open, write_open
from read.utils.table import linearize, linearize_tapex

# convert KILT datasets to dataloader_biencoder format
# the provenance for KILT answers has a wikipedia_id, and start/end paragraph ids, we need to map these to a set of overlapping passages
"""
{"id": "935392b3-c206-4036-94b0-09bc35319c45",
"input": "Cirith Ungol [SEP] genre",
"output": [{"answer": "heavy metal", "provenance": [{"wikipedia_id": "9065794", "title": "King of the Dead (album)", "section": "Section::::Abstract.", "start_paragraph_id": 1, "start_character": 84, "end_paragraph_id": 1, "end_character": 161, "bleu_score": 1.0}]},
{"answer": "heavy metal", "provenance": [{"wikipedia_id": "9065986", "title": "One Foot in Hell", "section": "Section::::Abstract.", "start_paragraph_id": 1, "start_character": 153, "end_paragraph_id": 1, "end_character": 266, "bleu_score": 1.0}]},
{"answer": "Heavy Metal"}, {"answer": "metal"}, {"answer": "heavy metal music"}, {"answer": "Metal"}, {"answer": "Metal music"}, {"answer": "Heavy Metal Music"}], "meta": {"subj_aliases": [], "sub_surface": ["Cirith Ungol"], "obj_surface": ["Heavy metal", "Heavy Metal", "heavy metal music", "heavy metal", "heavy-metal", "heavy-metal music", "hard rock", "traditional heavy metal", "thrash/black metal", "heavy rock", "Heavy Metal Music"]}}

{"id": "a64179cb-01fd-42e9-9f40-bf3f0a1ad1e4",
"input": "USA-88 [SEP] time of spacecraft launch",
"output": [{"answer": "3 February 1993", "provenance": [{"wikipedia_id": "36387881", "title": "USA-88", "start_paragraph_id": 2, "start_character": 0, "end_paragraph_id": 2, "end_character": 145, "bleu_score": 1.0, "meta": {}, "section": "Section::::Abstract."}]}],
"meta": {"template_questions": ["What is the launch date of USA-88?", "What day was USA-88 launched?", "When was the launch date of USA-88?", "When was USA-88's launch date?", "On what date did USA-88 take off?", "What date was USA-88 launched?", "What was the launch date of USA-88?", "What was the date of USA-88's launch?", "On what date did USA-88 launch?", "On what date was USA-88 launched?"]}}
"""

# first load passage_ids.txt, create map doc_id -> passage_ids


def matching_passage_ids(pids, start_para, end_para):
    def overlap(pas_start, pas_end):
        return (min(pas_end, end_para) - max(pas_start, start_para)) / (
            end_para - start_para
        )

    pid_with_overlap = [
        (pid, overlap(start, end))
        for pid, start, end in pids
        if start < end_para and end > start_para
    ]
    return pid_with_overlap


def process(
    line, cfg, title2doc_id, doc_id2pids, pid2passage, passage_count_distribution
):
    jobj = json.loads(line)
    inst_id = jobj["example_id"]
    if cfg.linearize == "normal":
        input = f"title : {jobj['table_page_title']} ; " + f"section : {jobj['table_section_title']} ; " + linearize(json.loads(jobj["table"]), jobj["highlighted_cells"])
    elif cfg.linearize == "tapex":
        input = linearize_tapex(json.loads(jobj["table"]), jobj["highlighted_cells"])
    else:
        input = jobj["subtable_metadata_str"]
    df = pd.DataFrame(json.loads(jobj["table"]))
    bm25_query = " ".join([df.iloc[i, j] for i, j in jobj["highlighted_cells"]])
    answers = [o["original_sentence"] for o in jobj["sentence_annotations"]]
    if len(answers) == 0:
        print(f"WARNING: no answers: {line}")
    pid_with_overlap, positives = get_overlap_and_positive(
        cfg, jobj, title2doc_id, doc_id2pids, pid2passage
    )
    pcount = (
        len(positives)
        if len(positives) < len(passage_count_distribution)
        else len(passage_count_distribution) - 1
    )
    passage_count_distribution[pcount] += 1
    if len(pid_with_overlap) == 0:
        return None
    jobj.update(
        {
        "id": inst_id,
        "query": input,
        "bm25_query": bm25_query,
        "answers": answers,
        "positive_pids": positives,
        "overlap_pids": pid_with_overlap,
        "positive_passages": [pid2passage[pid] for pid in positives],
        }
    )
    return jobj

def get_overlap_and_positive(cfg, jobj, title2doc_id, doc_id2pids, pid2passage):
    pid_with_overlap = []
    doc_id = title2doc_id.get(jobj["table_page_title"], None)
    if doc_id is None:
        return [], []

    for annotation in jobj["sentence_annotations"]:
        sentence = annotation["original_sentence"]

        for pid, start_para, end_para in doc_id2pids[doc_id]:
            passage = pid2passage[pid]
            if sentence[1:-1] in passage["text"]:
                pid_with_overlap.extend(
                    matching_passage_ids(doc_id2pids[doc_id], start_para, end_para)
                )
    pid_with_overlap = list(set(pid_with_overlap))
    if len(pid_with_overlap) == 0:
        return pid_with_overlap, []
    pid_with_overlap.sort(key=lambda x: x[1], reverse=True)
    min_overlap = min(cfg.min_overlap, pid_with_overlap[0][1])
    positives = list(set([pid for pid, o in pid_with_overlap if o >= min_overlap]))
    return pid_with_overlap, positives


@hydra.main(
    version_base="1.2", config_path=root / "config" / "dpr", config_name="dpr_generate_positives"
)
def main(cfg):
    # get the knowledge souce
    data_dir = Path(cfg.data_dir)
    # get pages by title
    logger.info("Mapping documents to passage ids")
    doc_id2pids  = json.load(open(cfg.doc_id2pids_file, "r"))
    title2doc_id = json.load(open(cfg.title2doc_id_file, "r"))
    doc_id2title = {v: k for k, v in title2doc_id.items()}
    pids = set()

    for doc_id in doc_id2pids:
        if doc_id in doc_id2title:
            pids.update(set([x[0] for x in doc_id2pids[doc_id]]))


    logger.info("Loading passage data")
    if not Path(cfg.pid2passages_file).exists():
        pid2passage = dict()
        passages_dir = Path(cfg.passages)

        for file in passages_dir.iterdir():
            for line in read_open(file).readlines():
                jobj = json.loads(line)
                pid = jobj['pid']
                if pid in pids:
                    pid2passage[pid] = jobj
        json.dump(pid2passage, open(cfg.pid2passages_file, 'w'))
    pid2passage = json.load(open(cfg.pid2passages_file, "r"))

    logger.info("Generating data")
    total = 0
    non_provenance = 0
    passage_count_distribution = np.zeros(6, dtype=np.float)

    for data_file in list(data_dir.iterdir()):
        if data_file.is_dir():
            continue
        output_file = data_dir / cfg.output_dir / data_file.name
        f = partial(process, cfg=cfg, title2doc_id=title2doc_id, doc_id2pids=doc_id2pids, pid2passage=pid2passage, passage_count_distribution=passage_count_distribution)
        with write_open(output_file) as writer:

            with read_open(data_file) as reader:
                lines = [str(x) for x in reader.readlines()]
                print("length", len(lines))
                logger.info("Start processing {}".format(data_file))
                for line in lines:
                    total += 1
                    try:
                        result = f(line) 
                    except KeyError as e:
                        print(e)
                        break
                    if result is not None:
                        writer.write(json.dumps(result) + "\n")
                    else:
                        non_provenance += 1
                    if total % 10000 == 0:
                        logger.info(f'{data_file} instances with no passage {non_provenance} out of {total}')
            logger.info("Finish processing {}".format(data_file))

    print(passage_count_distribution)
    # display distribution of number of positive passages
    passage_count_distribution *= 1.0 / passage_count_distribution.sum()
    percentages = [
        (f"{c}: " if c < len(passage_count_distribution) - 1 else f">={c}: ")
        + ("<1%" if 0.0 < p < 0.01 else f"{int(round(100*p))}%")
        for c, p in enumerate(passage_count_distribution)
    ]
    logger.info(f'Passage counts: {"; ".join(percentages)}')


if __name__ == "__main__":
    main()
