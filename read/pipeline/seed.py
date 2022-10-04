import pyrootutils
from typing import List
from kilt.knowledge_source import KnowledgeSource
from blingfire import text_to_sentences
from typing import Optional
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
from tango import Step, JsonFormat, Format

@Step.register("seed_document_retrieval")
class DocumentRetrieval(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    @staticmethod
    def init_searcher(searcher, faiss_index, lucene_index, qry_encoder, ctx_encoder):
        if searcher == "dpr":
            searcher=DPRSearcher(faiss_index, lucene_index, qry_encoder)
        elif searcher == "hybrid":
            searcher=HybridSearcher(lucene_index, qry_encoder, ctx_encoder)

    def run(self, data: List[dict], searcher, faiss_index, lucene_index, qry_encoder, ctx_encoder, batch_size) -> List[bool]:
        searcher = self.init_searcher(searcher, faiss_index, lucene_index, qry_encoder, ctx_encoder)

        hits_per_query = []
        batch = []
        
        for example in data: 
            batch.append(example)
            if len(batch) == batch_size:
                hits_per_query += searcher.search(batch, k=10)
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

                page = self.ks.get_page_by_id(id)
                if page is None:
                    continue
                for passage in page["text"]:
                    if "::::" in passage:
                        continue
                    for sent in text_to_sentences(passage).split("\n"):
                        if len(sent.split()) <= 4: # Short sentences are mostly section titles. 
                            continue                   
                        doc_result.append((sent, hit["score"]))
            doc_results.append(doc_result)
        return doc_results


@Step.register("seed_sentence_selection")
class SenteceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, model, tokenizer, data, doc_results: List[List[str]]) -> List[bool]:
        sent_selection = SentenceSelector(model, tokenizer)

        sentence_results = []
        for i, (example, doc_result) in enumerate(zip(data, doc_results)):
            batch = []
            selected_sents = []
            for sent, score in doc_result:
                batch.append((example, sent))
                selected_mask = sent_selection(*zip(*batch))
                for example, selected in zip(batch, selected_mask):
                    if selected:
                        selected_sents.append(sent)
                sentence_results.append(selected_sents)
        return sentence_results


@Step.register("seed_table_verification")
class TableVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self,model, tokenizer, data, sentence_results: List[List[str]]) -> List[bool]:
        verifier=TableVerifier(model, tokenizer)

        verified_results = []
        for i, (example, result) in enumerate(zip(data, sentence_results)):
            verified_result = []
            for sent in result:
                if verifier.verify(sent, example["table"]):
                    verified_result.append(True)
                    break
            else:
                verified_result.append(False)
        return verified_results


@Step.register("seed_error_correction")
class ErrorCorrection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, examples, sentence_results: List[List[str]]) -> List[bool]:
        corrected_results = []
        for i, (example, result) in enumerate(zip(examples, sentence_results)):
            corrected_result = []
            for sent in result:
                corrected_result.append(self.verifier.correct(sent, example["table"]))
        return corrected_results



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
        
        result = []
        for i, (table, sent, idx) in enumerate(table_with_sent):
            result.append({"table": examples[idx]["table"], "sentence": sent, "id": examples[idx]["id"], "title": examples[idx]["title"], "label": 0})

        return result

                

