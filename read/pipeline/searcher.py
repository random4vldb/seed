import json
from typing import List

import numpy as np
import torch
from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)
import pandas as pd
from loguru import logger
from tango import Step, Format, JsonFormat
from typing import Optional
from kilt.knowledge_source import KnowledgeSource

from blingfire import text_to_sentences


class DPRSearcher:
    def __init__(self, faiss_index, lucene_index, qry_encoder):
        logger.info("Loading DPR model")
        self.searcher = FaissSearcher(faiss_index, qry_encoder)
        self.searcher.ssearcher = LuceneSearcher(lucene_index)

    def process_content(self, content):
        content = json.loads(content)
        title, doc = content["title"], content["contents"]
        return title, doc

    def process_result(self, hits, include_docs=False, include_vectors=False):
        result = []
        scores = []
        vectors = []
        for hit in hits:
            docid = hit.docid
            score = hit.score
            doc = self.searcher.doc(docid)
            scores.append(score)
            if include_docs:
                title, doc = self.process_content(doc.raw())
                result.append(
                    {"title": title, "text": doc, "score": score, "id": docid}
                )
            else:
                result.append({"score": score, "id": docid, "title": title})
            if include_vectors:
                result[-1]["vectors"] = hit.vectors
                vectors.append(hit.vectors)
        if include_vectors:
            return result, scores, vectors
        return result, scores

    def query(self, query, k, include_docs=True, include_vectors=False):
        hits = self.searcher.search(query, k, return_vector=include_vectors)

        if include_vectors:
            _, hits = hits

        return self.process_result(hits, include_docs, include_vectors)

    def batch_query(
        self, queries, k=10, num_threads=1, include_docs=True, include_vectors=False
    ):
        result = self.searcher.batch_search(
            queries,
            [str(x) for x in list(range(len(queries)))],
            threads=num_threads,
            k=k,
            return_vector=include_vectors,
        )

        if include_vectors:
            _, result = result

        outputs = [[] for _ in range(len(queries))]
        scores = [[] for _ in range(len(queries))]
        vectors = [[] for _ in range(len(queries))]
        for qid, hits in result.items():

            if include_vectors:
                (
                    outputs[int(qid)],
                    scores[int(qid)],
                    vectors[int(qid)],
                ) = self.process_result(hits, include_docs, include_vectors)
            else:
                outputs[int(qid)], scores[int(qid)] = self.process_result(
                    hits, include_docs, include_vectors
                )

        if include_vectors:
            return outputs, scores, vectors
        return outputs, scores

    def __call__(self, examples, k=10):
        outputs, _ = self.batch_query(
            [x["linearized_table"] for x in examples], k, include_docs=True
        )

        return outputs



class HybridSearcher:
    def __init__(self, lucene_index, qry_encoder_path, ctx_encoder_path):
        self.searcher = LuceneSearcher(lucene_index)
        (
            self.qry_tokenizer,
            self.qry_encoder,
            self.ctx_tokenizer,
            self.ctx_encoder,
        ) = self._get_tokenizer_and_model(qry_encoder_path, ctx_encoder_path)

    def _get_tokenizer_and_model(self, qry_encoder_path, ctx_encoder_path):
        qry_encoder = DPRQuestionEncoder.from_pretrained(qry_encoder_path)
        qry_encoder.eval()
        ctx_encoder = DPRContextEncoder.from_pretrained(ctx_encoder_path)
        ctx_encoder.eval()
        qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-question_encoder-multiset-base"
        )
        ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(
            "facebook/dpr-ctx_encoder-multiset-base"
        )
        return qry_tokenizer, qry_encoder, ctx_tokenizer, ctx_encoder

    def ctx_embed(self, doc_batch: List[dict]) -> np.ndarray:
        documents = {
            "title": [doci["title"] for doci in doc_batch],
            "text": [doci["contents"] for doci in doc_batch],
        }
        """Compute the DPR embeddings of document passages"""
        input_ids = self.ctx_tokenizer(
            documents["title"],
            documents["text"],
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )["input_ids"]

        with torch.no_grad():
            embeddings = self.ctx_encoder(
                input_ids.to(device=self.ctx_encoder.device), return_dict=True
            ).pooler_output
        return embeddings.detach().cpu().to(dtype=torch.float16).numpy()

    def qry_embed(self, qry_batch: List[str]) -> np.ndarray:
        inputs = self.qry_tokenizer(
            qry_batch, truncation=True, padding="longest", return_tensors="pt"
        )
        with torch.no_grad():
            embeddings = self.qry_encoder(
                inputs["input_ids"].to(device=self.qry_encoder.device),
                inputs["attention_mask"].to(device=self.qry_encoder.device),
                return_dict=True,
            ).pooler_output
        return embeddings.detach().cpu().to(dtype=torch.float16).numpy()

    def process_result(self, query, hits, include_docs=False):
        hits = [json.loads(hit.raw) for hit in hits]
        ctx_vecs = [self.ctx_embed([hit]).reshape(-1) for hit in hits]
        qry_vec = self.qry_embed([query]).reshape(-1)

        scores = [(np.dot(qry_vec, ctx_veci)) for p, ctx_veci in zip(hits, ctx_vecs)]

        if include_docs:
            result = [
                {
                    "title": hit["title"],
                    "text": hit["contents"],
                    "score": score,
                    "id": hit["id"],
                }
                for hit, score in zip(hits, scores)
            ]
        else:
            result = [
                {"title": hit["title"], "score": score, "id": hit["id"]}
                for hit, score in zip(hits, scores)
            ]

        return result, scores

    def query(self, query, k, include_docs=True, include_vectors=False):
        hits = self.searcher.search(query, k * 2)

        result, scores = self.process_result(hits)

        result = sorted(result, key=lambda x: x["score"], reverse=True)[:k]
        scores = sorted(scores, reverse=True)[:k]

        return result, scores

    def batch_query(
        self, queries, k=10, num_threads=1, include_docs=True, include_vectors=False
    ):
        result = self.searcher.batch_search(
            queries,
            [str(x) for x in list(range(len(queries)))],
            threads=num_threads,
            k=k,
        )

        outputs = [[] for _ in range(len(queries))]
        scores = [[] for _ in range(len(queries))]
        for qid, hits in result.items():
            output, score = self.process_result(queries[int(qid)], hits, include_docs)

            outputs[int(qid)] = sorted(output, key=lambda x: x["score"], reverse=True)[
                :k
            ]
            scores[int(qid)] = sorted(score, reverse=True)[:k]

        return outputs, scores

    def __call__(self, examples, k=10):
        queries = []
        for example in examples:
            table = pd.DataFrame(example["table"])
            query = " ".join(
                [table.iloc[i, j] for i, j in example["highlighted_cells"]]
            )
            query = example["title"] + " " + example["title"] + " " + query
            queries.append(query)

        outputs, _ = self.batch_query(
            queries, k, include_docs=True
        )

        return outputs


@Step.register("pipeline::document_retrieval")
class DocumentRetrieval(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0012"

    @staticmethod
    def init_searcher(searcher, faiss_index, lucene_index, qry_encoder, ctx_encoder):
        if searcher == "dpr":
            searcher = DPRSearcher(faiss_index, lucene_index, qry_encoder)
        elif searcher == "hybrid":
            searcher = HybridSearcher(lucene_index, qry_encoder, ctx_encoder)

        return searcher

    def run(
        self,
        data: List[dict],
        searcher,
        faiss_index,
        lucene_index,
        qry_encoder,
        ctx_encoder,
        batch_size,
    ) -> List[bool]:
        searcher = self.init_searcher(
            searcher, faiss_index, lucene_index, qry_encoder, ctx_encoder
        )

        ks = KnowledgeSource()
        hits_per_query = []
        batch = []

        for example in data:
            batch.append(example)
            if len(batch) == batch_size:
                hits_per_query += searcher(batch, k=10)
                batch = []

        doc_results = []

        for i, (example, hits) in enumerate(zip(data, hits_per_query)):
            doc_result = []
            ids = set()
            for hit in hits:
                id = hit["id"].split("::")[0]

                if id in ids:
                    continue
                else:
                    ids.add(id)

                page = ks.get_page_by_id(id)
                if page is None:
                    continue
                for passage in page["text"]:
                    if "::::" in passage:
                        continue
                    for sent in text_to_sentences(passage).split("\n"):
                        if (
                            len(sent.split()) <= 4
                        ):  # Short sentences are mostly section titles.
                            continue
                        doc_result.append((sent, hit["score"].item()))
            doc_results.append(doc_result)
        return doc_results
