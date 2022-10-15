from tango import Step, JsonFormat, Format
from typing import Optional


@Step.register("infotab::pipeline_preprocess")
class InfotabPipelinePreProcess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, data, doc_results):
        examples = []
        for idx, (example, doc_result) in enumerate(zip(data, doc_results)):
            for sentence in doc_result:
                examples.append(
                    {
                        "table_id": idx,
                        "annotator_id": idx,
                        "hypothesis":  sentence,
                        "table": example["table"],
                        "title": example["title"],
                        "highlighted_cells": example["highlighted_cells"],
                    }
                )

        return examples

@Step.register("infotab::pipeline_postprocess")
class InfotabPipelinePostProcess(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, results):
        id2results = {}
        for idx, result in enumerate(results):
            if result["table_id"] in id2results:
                if results["label"] == 1:
                    id2results[result["table_id"]] = True
                else:
                    id2results[result["table_id"]] = False
            else:
                if results["label"] == 1:
                    id2results[result["table_id"]] = True
                
        final_results = [False] * len(results)
        for idx, result in id2results.items():
            final_results[idx] = id2results[idx]
        
        return final_results



@Step.register("infotab::document_retrieval")
class InfotabSentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, data, sentence_results):
        for example, sentence_result in zip(data, sentence_results):
            for doc, score, title in sentence_result:
                if title == example["title"]:
                    example["doc"] = doc
                    example["score"] = score
                    break
        