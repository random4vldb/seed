import functools
import json
import logging
import multiprocessing

import hydra
import jsonlines
import pyrootutils
from loguru import logger
from pyserini.search.lucene import LuceneSearcher


def to_distinct_doc_ids(passage_ids):
    doc_ids = set()
    for pid in passage_ids:
        doc_id = pid[:pid.find(':')]
        doc_ids.add(doc_id)
    return list(doc_ids)


def _retrieve_one(query_exclude, searcher, k):
    query, exclude = query_exclude
    hits = searcher.search(query, k)
    docs = {"pid": ["N/A:0"] * k, "title": ["title"] * k, "text": ["text"] * k}
    doc_scores = [0.0] * k
    titles = []
    texts = []
    if len(hits) == 0:
        # create dummy docs if no result
        logger.warning(f"No results for {query}!")
        return doc_scores, docs
    if len(hits) < k:
        # duplicate last doc if too few results
        logger.warning(f"Too few results for {query}! ({len(hits)})")
        hits.extend([hits[-1]] * (k - len(hits)))
    for hit in hits:
        content = json.loads(hit.raw)

        assert len(hits) == k
        doc_scores = [hit.score for hit in hits]
        title, text = content["title"], content["contents"]
        titles.append(title)
        texts.append(text)
    docs = {"pid": [hit.docid for hit in hits], "title": titles, "text": texts}
    return doc_scores, docs


class BM25:
    def __init__(
        self,
        cfg
    ):
        """
        :param hypers:
        """
        self.cfg = cfg
        self.searcher = LuceneSearcher(cfg.index_path)
        if cfg.num_processes > 1:
            # NOTE: only thread-based pooling works with the JSearcher
            self.pool = multiprocessing.pool.ThreadPool(processes=cfg.num_processes)
            logger.info(f"Using multiprocessing pool with {cfg.num_processes} workers")
        else:
            self.pool = None
        self._retrieve_one = functools.partial(
            _retrieve_one,
            searcher=self.searcher,
            k=cfg.num_answers
        )

    def retrieve_forward(self, queries, *, exclude_by_content=None):
        """

        :param queries: list of queries to retrieve documents for
        :return: input for RAG: context_input_ids, context_attention_mask, doc_scores
          also docs and info-for-backward (when calling retrieve_backward)
        """
        if exclude_by_content is None:
            exclude_by_content = [set() for _ in range(len(queries))]
        else:
            assert len(exclude_by_content) == len(queries)

        if self.pool is not None:
            result_batch = self.pool.map(
                self._retrieve_one, zip(queries, exclude_by_content)
            )
            docs = [r[1] for r in result_batch]
            doc_scores = [r[0] for r in result_batch]
        else:
            docs = []
            doc_scores = []
            for query, exclude in zip(queries, exclude_by_content):
                doc_scores_i, docs_i = self._retrieve_one((query, exclude))
                doc_scores.append(doc_scores_i)
                docs.append(docs_i)

        return doc_scores, docs

    def retrieve(self, queries):
        doc_scores, docs = self.retrieve_forward(queries)
        
        if 'id' in docs[0]:
            retrieved_doc_ids = [dd['id'] for dd in docs]
        elif 'pid' in docs[0]:
            retrieved_doc_ids = [dd['pid'] for dd in docs]
        else:
            retrieved_doc_ids = [['0:0'] * len(dd['text']) for dd in docs]  # dummy ids
        passages = None
        if self.cfg.include_passages:
            passages = [{'titles': dd['title'], 'texts': dd['text']} for dd in docs]
        assert type(retrieved_doc_ids) == list
        assert all([type(doc_ids) == list for doc_ids in retrieved_doc_ids])
        if not all([type(doc_id) == str for doc_ids in retrieved_doc_ids for doc_id in doc_ids]):
            print(f'Error: {retrieved_doc_ids}')
            raise ValueError('not right type')
        return retrieved_doc_ids, passages, doc_scores


    def record_one_instance(self, inst_id, input, dpr_query, doc_ids, passages, scores):
        wids = to_distinct_doc_ids(doc_ids)
        pred_record = {'id': inst_id, 'bm25_query': input, "query": dpr_query,  'output': [{'answer': '', 'provenance': [{'wikipedia_id': wid} for wid in wids]}]}
        if passages:
            pred_record['passages'] = [{'pid': pid, 'title': title, 'text':text} for pid, title, text in zip(doc_ids, passages['titles'], passages['texts'])]
        pred_record["scores"] = scores
        return pred_record


    def one_batch(self, id_batch, query_batch, dpr_batch):
        """
        retrieve and record one batch of queries
        :param id_batch:
        :param query_batch:
        :param output:
        :return:
        """
        records = []
        retrieved_doc_ids, passages, scores = self.retrieve(query_batch)
        for bi in range(len(query_batch)):
            records.append(self.record_one_instance(id_batch[bi], query_batch[bi], dpr_batch[bi],  retrieved_doc_ids[bi], passages[bi] if passages else None, scores[bi]))
        return records



