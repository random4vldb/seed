from copy import deepcopy
from transformers import (DPRQuestionEncoder, RagTokenizer, RagTokenForGeneration)
import torch
import ujson as json
import jsonlines
import pyrootutils
from loguru import logger
from pathlib import Path
import pprint
import hydra

from read.utils.report import Report

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import write_open
from read.metrics.retrieval import DocRetrievalEvaluator
from read.dpr.searcher import DPRSearcher

torch.set_grad_enabled(False)

# support loading from either a rag_model_path or qry_encoder_path

@hydra.main(version_base="1.2", config_path=root / "config" / "eval", config_name="dpr_search.yaml")
def main(cfg):
    searcher = DPRSearcher(cfg.faiss_index, cfg.lucene_index, cfg.qry_encoder_path)
    predictions = []
    golds = []
    report = Report(check_every=10)
    retrieval_evaluator = DocRetrievalEvaluator(cfg.ks)

    with jsonlines.open(cfg.data) as reader:        
        queries = []

        for jobj in list(reader):
            golds.append(deepcopy(jobj))
            queries.append(jobj["query"])
            if len(queries) == 10:
                outputs, _ = searcher.batch_query(queries, k=cfg.num_answers, include_vectors=False, include_docs=True, num_threads=cfg.num_processes)
                queries = []
                for output in outputs:
                    result = {}
                    result['passages'] = [{"pid": hit["id"],"text": hit["text"]} for hit in output]
                    
                    result["scores"] = [hit["score"].tolist() for hit  in output]
                    predictions.append(result)

                    if report.is_time():
                        logger.info(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. ')

            if (len(predictions) + 1) % 100 == 0:
                logger.info(f"Evaluating for {len(predictions) + 1} instances")
                result = retrieval_evaluator.evaluate(predictions, golds)
                pprint.pprint(result)

        
        if len(queries) > 0:
            outputs, _ = searcher.batch_query(queries, k=cfg.num_answers, include_vectors=False, include_docs=True, num_threads=cfg.num_processes)
            for output in outputs:
                result = {}

                result['passages'] = [{"pid": hit["id"],"text": hit["text"]} for hit in output]

                result["scores"] = [hit["score"].tolist() for hit  in output]
                predictions.append(result)

                if report.is_time():
                    logger.info(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. ')
        logger.info(f'Finished instance {report.check_count}; {report.check_count/report.elapsed_seconds()} per second. ')
    # now do evaluation

    logger.info("Evaluating")
    retrieval_evaluator = DocRetrievalEvaluator(cfg.ks)
    result = retrieval_evaluator.evaluate(predictions, golds)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(cfg.output, "w") as writer:
        writer.write_all(predictions)
    pprint.pprint(result)


if __name__ == "__main__":
    main()