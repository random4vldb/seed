from collections import defaultdict
from pathlib import Path

import hydra
import jsonlines

import pyrootutils
import ujson as json
from kilt.knowledge_source import KnowledgeSource
from loguru import logger
from multiprocessing import Pool
from tqdm import tqdm

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


@hydra.main(
    version_base="1.2", config_path=root / "config", config_name="dpr_filter.yaml"
)
def main(cfg):
    # get the knowledge souce
    logger.info("Mapping documents to passage ids")
    if not Path(cfg.doc_id2pids_file).exists():
        data_dir = Path(cfg.data_dir)
        doc_id2pids = defaultdict(list)
        for line in open(cfg.passage_ids, "r"):
            line = line.strip()
            doc_id_end = line.find("::")
            doc_id = line[:doc_id_end]
            range = line[doc_id_end + 3 : -1]
            range_open = line[doc_id_end + 2]
            range_end = line[-1]
            start, end = [float(r) for r in range.split(",")]
            if range_open == "(":
                start += 0.5
            else:
                assert range_open == "["
            if range_end == ")" and end > start:
                end += 0.5
            else:
                assert range_end == "]" or end == start
                end += 1.0
            doc_id2pids[doc_id].append((line, start, end))

        json.dump(doc_id2pids, open(cfg.doc_id2pids_file, "w"))

    # get pages by title
    logger.info("Processing documents")


    data_dir = Path(cfg.data_dir)
    title2id = {}
    titles = []
    for data_file in data_dir.iterdir():
        with jsonlines.open(data_file) as reader:
            objs = [obj for obj in reader]
            logger.info("Running multiple processing")
    
            titles.extend([x["table_page_title"] for x in objs])

    ks = KnowledgeSource()
    pages = ks.db.find({"wikipedia_title": {"$in": titles}})
            
    title2id = {x["wikipedia_title"]: x["wikipedia_id"] for x in pages}
    print(len(title2id))
    json.dump(title2id, open(cfg.title2id_file, "w"))


if __name__ == "__main__":
    main()
