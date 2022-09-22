from multiprocessing.pool import ThreadPool
import functools
import logging
import math
from pyserini.search import LuceneSearcher
import re
import pandas as pd
import pyrootutils
import hydra
import jsonlines
import json
from loguru import logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import write_open
from read.utils.report import Report
from read.utils.text import text_normalize


def _retrieve_one(
    query_tuple, searcher, negatives_per_positive, max_instance_expansion
):
    inst_id, query, dpr_query, positive_pids, positive_context, answers = query_tuple
    normed_positives = [
        " " + p["title"] + "  " + p["text"] + " "
        for p in positive_context
    ]
    target_num_negatives = negatives_per_positive * min(
        max_instance_expansion, len(positive_pids)
    )
    # initially fetch a factor of 3 more than we need, since we filter out some
    hits = searcher.search(query, 3 * target_num_negatives)
    negs = []
    for hit in hits:
        pid = hit.docid
        content = json.loads(hit.raw)
        # CONSIDER: if max_bm25_score > 0 and hit.score >= max_bm25_score: continue
        if pid in positive_pids:
            continue
        title, text = content["title"], content["contents"]
        norm_context = " " + title + " " + text + " "
        if norm_context in normed_positives:
            continue  # exclude any passage with the same text as a positive from negatives
        answer_bearing = any([ans in norm_context for ans in answers])
        if answer_bearing:
            continue  # exclude answer bearing passages from negatives
        negs.append({"pid": pid, "title": title, "text": text})
        if len(negs) >= target_num_negatives:
            break
    return negs


class BM25forDPR:
    def __init__(self, index: str, cfg):
        self.searcher = LuceneSearcher(index)
        self.cfg = cfg
        # NOTE: only thread-based pooling works with the JSearcher
        self.pool = ThreadPool(cfg.num_processes)
        logger.info(f"Using multiprocessing pool with {cfg.num_processes} workers")
        self.no_negative_skip_count = 0
        self._retrieve_one = functools.partial(
            _retrieve_one,
            searcher=self.searcher,
            negatives_per_positive=cfg.max_negatives,
            max_instance_expansion=cfg.max_instance_expansion,
        )
        self.written = 0


    def _write_batch(self, out, query_tuples, all_negs):
        for query_tuple, negs in zip(query_tuples, all_negs):
            inst_id, query, dpr_query, positive_pids, positive_context, answers = query_tuple

            if len(negs) == 0:
                if self.no_negative_skip_count == 0:
                    logger.warning(f'No negatives for "{query}"\n   Answers: {answers}')
                self.no_negative_skip_count += 1
                continue

            num_instances = min(
                len(positive_context),
                self.cfg.max_instance_expansion,
                int(math.ceil(len(negs) / self.cfg.max_negatives)),
            )

            for pndx, pos in enumerate(positive_context[:num_instances]):
                out.write(
                    json.dumps(
                        {
                            "id": inst_id,
                            "query": dpr_query,
                            "positive": pos,
                            "negatives": negs[pndx::num_instances][
                                : self.cfg.max_negatives
                            ],
                        }
                    )
                    + "\n"
                )
                self.written += 1

    def create(self, positive_pids_file, output_dir):
        report = Report()
        batch_size = 1024
        with write_open(output_dir) as out:
            query_tuples = []
            with jsonlines.open(positive_pids_file) as reader:
                for jobj in reader:
                    inst_id = jobj["id"]
                    query = jobj["bm25_query"]
                    dpr_query = jobj["query"]
                    positive_pids = jobj["positive_pids"]
                    positive_context = jobj["positive_passages"]
                    if self.cfg.allow_answer_bearing:
                        answers = []
                    else:
                        answers = [a.strip() for a in jobj["answers"]]
                        answers = [" " + a + " " for a in answers if a]
                    query_tuples.append(
                        (
                            inst_id,
                            query,
                            dpr_query,
                            positive_pids,
                            positive_context,
                            answers,
                        )
                    )
                    if len(query_tuples) >= batch_size:
                        all_negs = self.pool.map(self._retrieve_one, query_tuples)
                        self._write_batch(out, query_tuples, all_negs)
                        query_tuples = []
                        if report.is_time():
                            instance_count = report.check_count * batch_size
                            logger.info(
                                f"On instance {instance_count}, "
                                f"{instance_count/report.elapsed_seconds()} instances per second"
                            )
                            if self.no_negative_skip_count > 0:
                                logger.info(
                                    f"{self.no_negative_skip_count} skipped for lack of negatives"
                                )

                if len(query_tuples) > 0:
                    all_negs = self.pool.map(self._retrieve_one, query_tuples)
                    self._write_batch(out, query_tuples, all_negs)
                instance_count = report.check_count * batch_size
                logger.info(
                    f"Finished {instance_count} instances; wrote {self.written} training triples. "
                    f"{instance_count/report.elapsed_seconds()} instances per second"
                )
                if self.no_negative_skip_count > 0:
                    logger.info(
                        f"{self.no_negative_skip_count} skipped for lack of negatives"
                    )


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "dpr",
    config_name="dpr_generate_training.yaml",
)
def main(cfg):
    bm25dpr = BM25forDPR(cfg.pyserini_index, cfg)
    bm25dpr.create(cfg.positive_pids, cfg.output_dir)


if __name__ == "__main__":
    main()
