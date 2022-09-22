import jsonlines
import pyrootutils
import hydra
import json
import pprint
from pathlib import Path

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


from read.dpr.baselines.bm25 import BM25
from read.metrics.retrieval import DocRetrievalEvaluator


@hydra.main(version_base="1.2", config_path=root / "config" / "eval", config_name="bm25.yaml")
def main(cfg):
    bm = BM25(cfg)
    predictions = []
    id_batch, query_batch, dpr_batch = [], [], []
    with jsonlines.open(cfg.data, "r") as reader:
        for line_ndx, inst in enumerate(reader):
            if 0 < cfg.instance_limit <= line_ndx:
                break
            id_batch.append(inst['id'])
            query_batch.append(inst["table_page_title"] + " " + inst["table_section_title"] + " "  + inst['bm25_query'])
            dpr_batch.append(inst['query'])
            if len(query_batch) == 2 * cfg.num_processes:
                predictions.extend(bm.one_batch(id_batch, query_batch, dpr_batch))
                id_batch, query_batch, dpr_batch = [], [], []
        if len(query_batch) > 0:
            predictions.extend(bm.one_batch(id_batch, query_batch, dpr_batch))

    golds = []
    with jsonlines.open(cfg.data) as reader:
        for jobj in reader:
            golds.append(jobj)
    # now do evaluation

    retrieval_evaluator = DocRetrievalEvaluator(cfg.ks)
    result = retrieval_evaluator.evaluate(predictions, golds)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(cfg.output, "w") as writer:
        writer.write_all(predictions)

    pprint.pprint(result)

if __name__ == "__main__":
    main()