from .seed import SEEDPipeline
from loguru import logger

class InfotabPipeline(SEEDPipeline):
    def __init__(self, searcher, sent_selection, verifier, evaluator) -> None:
        super().__init__(searcher, sent_selection, verifier, evaluator)

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
                    table_with_sent.append({
                        "id": id,
                        "title": hit["title"],
                        "passage": passage,
                        "label": example["label"],
                        "query": example["query"]
                    })
                if hit["title"] == examples[i]["title"]:
                    table_with_sent.append({
                        "id": id,
                        "title": hit["title"],
                        "passage": examples[i]["correct_sentence"],
                        "label": example["label"],
                        "query": example["query"]
                    })


        logger.info(f"Found {len(table_with_sent)} table passages")
        logger.info(f"Found {len(doc_results)} doc passages")

        sent_results = self.sent_selection(table_with_sent)
        sent_results = [x for x in sent_results if x is not None]

        logger.info(f"Found {len(sent_results)} sentences")

        sent_results = self.verifier(sent_results)

        logger.info(f"Found {len(sent_results)} sentences with a verdict")

        self.evaluator.add_results(sent_results)

        return [x["verdict"] for x in sent_results]
