from tango import Step, JsonFormat, Format
from typing import Optional
from typing import List

@Step.register("infotab_table_verification")
class InfotabTableVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, sent_selection_models, verification_models, tokenizer, data, sentence_results: List[List[str]]) -> List[bool]:
        pass


@Step.register("infotab::pipeline_preprocess")
class InfotabPipelinePreProcess(Step):
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