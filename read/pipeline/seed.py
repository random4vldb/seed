import pyrootutils
from typing import List
from kilt.knowledge_source import KnowledgeSource
from blingfire import text_to_sentences

from read.seed.verification.verifier import TableVerifier


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


from read.dpr.searcher import DPRSearcher, HybridSearcher
from read.seed.sent_selection import SentenceSelector
from read.metrics.pipeline import PipelineModule


class SEEDPipeline:
    def __init__(self, searcher, sent_selection, verifier, evaluator) -> None:
        self.searcher = searcher
        self.sent_selection = sent_selection
        self.verifier = verifier
        self.ks = KnowledgeSource()
        self.evaluator = evaluator

    @staticmethod
    def init_with_config(cfg, evaluator):
        if cfg.searcher.type == "dpr":
            return SEEDPipeline(
                searcher=DPRSearcher(cfg.searcher.faiss_index, cfg.searcher.lucene_index, cfg.searcher.qry_encoder, cfg),
                sent_selection=SentenceSelector(cfg.sent_selection.model, cfg.sent_selection.tokenizer, cfg),
                verifier=TableVerifier(cfg.verifier.model, cfg.verifier.tokenizer, cfg),
                evaluator=evaluator
            )
        elif cfg.searcher.type == "hybrid":
            return SEEDPipeline(
                searcher=HybridSearcher(cfg.searcher.lucene_index, cfg.searcher.qry_encoder, cfg.searcher.ctx_encoder, cfg),
                sent_selection=SentenceSelector(cfg.sent_selection.model, cfg.sent_selection.tokenizer, cfg),
                verifier=TableVerifier(cfg.verifier.model, cfg.verifier.tokenizer, cfg),
                evaluator=evaluator
            )
        

    def __call__(self, examples: List[dict]) -> List[bool]:
        hits_per_query = self.searcher(examples, k=10)
        table_with_sent = []

        doc_results = []

        for i, (example, hits) in enumerate(zip(examples, hits_per_query)):
            ids = set()
            for hit in hits:
                id = hit["id"].split("::")[0]

                if id in ids:
                    continue
                else:
                    ids.add(id)

                doc_results.append((hit["score"], hit["title"] == examples[i]["title"], i))
                page = self.ks.get_page_by_id(id)
                if page is None:
                    continue
                for passage in page["text"]:
                    if "::::" in passage:
                        continue
                    for sent in text_to_sentences(passage).split("\n"):
                        if len(sent.split()) <= 4: # Short sentences are mostly section titles. 
                            continue                   
                        table_with_sent.append((example["linearized_table"], sent, i))

        self.evaluator.update(PipelineModule.DOCUMENT_RETRIEVAL, *zip(*doc_results))

        sent_results = []
        

        selected_mask = self.sent_selection(*zip(*table_with_sent))
        result = [False] * len(table_with_sent)
        for i, (table, sent, idx) in enumerate(table_with_sent):
            sent_results.append((selected_mask[i], sent == examples[idx]["sentence"], idx))
            if selected_mask[i]:
                res, score = self.verifier([table], [sent])
                if res:
                    result[i] = True

        self.evaluator.update(PipelineModule.SENTENCE_SELECTION, *zip(*sent_results))


        final_result = [False] * len(examples)
        table_results = []
        for i in range(len(result)):
            for j, ex in enumerate(examples):
                if ex["linearized_table"] == table_with_sent[i][0] and result[i] == True:
                    final_result[j] = True
                    table_results.append((True, bool(examples[j]["label"]), j))
                    break

        for i, result in enumerate(final_result):
            if result == False:
                table_results.append((False, bool(examples[i]["label"]), i))    
            else:
                table_results.append((True, bool(examples[i]["label"]), i))   
        self.evaluator.update(PipelineModule.TABLE_VERIFICATION, *zip(*table_results))
        
        return final_result


    def generate_nli_data(self, examples: List[str]) -> None:
        hits_per_query = self.searcher(examples, k=10)
        table_with_sent = []

        for i, (example, hits) in enumerate(zip(examples, hits_per_query)):
            ids = set()
            for hit in hits:
                id = hit["id"].split("::")[0]

                if id in ids:
                    continue
                else:
                    ids.add(id)


                page = self.ks.get_page_by_id(id)
                if page is None:
                    continue
                for passage in page["text"]:
                    if "::::" in passage:
                        continue
                    for sent in text_to_sentences(passage).split("\n"):
                        if len(sent.split()) <= 4: # Short sentences are mostly section titles. 
                            continue                   
                        table_with_sent.append((example["linearized_table"], sent, i))
        
        selected_mask = self.sent_selection(*zip(*table_with_sent))
        result = []
        for i, (table, sent, idx) in enumerate(table_with_sent):
            if selected_mask[i]:
                result.append({"table": examples[idx]["table"], "sentence": sent, "id": examples[idx]["id"], "title": examples[idx]["title"], "label": 0})

        return result

                

