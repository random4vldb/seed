from transformers import AutoTokenizer
from accelerate import Accelerator
from tango import Format, JsonFormat, Step
from transformers import AutoTokenizer
from blingfire import text_to_sentences


@Step.register("pipeline::table_verification")
class TableVerification(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0027"


    def run(self, model, tokenizer, data, sentence_results):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        accelerator = Accelerator()

        model, tokenizer= accelerator.prepare(model, tokenizer)

        print("Data", len(data))
        print("Sent", len(sentence_results))
        verified_results = []
        for i, (example, result) in enumerate(zip(data, sentence_results)):
            for sent in result:
                if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
                    table = pd.DataFrame(example["table"])
                else:
                    table = example["linearized_table"]
                
                inputs = tokenizer(table=table, queries=sent, return_tensors="pt", padding=True, truncation=True)

                inputs = {k: v.cuda() for k, v in inputs.items()}

                pred = model(**inputs).logits.argmax(dim=1).tolist()

                if pred[0] == 1:
                    verified_results.append(1)
                    break
            else:
                verified_results.append(0)
            print("Length", len(verified_results), i)

        return verified_results


@Step.register("pipeline::sentence_selection")
class SentenceSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "003"


    def run(self, model, tokenizer, data, doc_results):
        tokenizer_type = tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

        all_preds = []

        if "tapas" in tokenizer_type or "tapex" in tokenizer_type:
            queries = [pd.DataFrame(x['table']) for x in data]
        else:
            queries = [x['linearized_table'] for x in data]

        accelerator = Accelerator()

        model, tokenizer= accelerator.prepare(model, tokenizer)


        for table, doc_result in zip(queries, doc_results):
            preds = []
            for doc, score, title in doc_result:
                for sent in text_to_sentences(doc):
                    inputs = tokenizer(table=table, queries=doc, return_tensors="pt", padding=True, truncation=True)

                    inputs = {k: v.cuda() for k, v in inputs.items()}

                    pred = model(**inputs).logits.argmax(dim=1).tolist()

                    if pred[0] == 1:
                        preds.append(doc)
                    preds.append(doc)

            all_preds.append(preds)

        return all_preds



@Step.register("pipeline::evaluation")
class Evaluation(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "00493"

    step2name2metrics = {
        x : {
            "accuracy": evaluate.load("accuracy"),
            "f1": evaluate.load("f1"),
            "precision": evaluate.load("precision"),
            "recall": evaluate.load("recall"),
        }
        for x in ["sentence_selection", "table_verification"]
    }

    step2name2metrics["document_retrieval"] = {
        "retrieval": evaluate.load("trec_eval")
    }

    def jaccard_similarity(self, list1, list2):
        return len(set(list1).intersection(set(list2))) / len(set(list1).union(set(list2)))

    def process_sentence_selection(self, data, sentence_results):
        
        for example, result in zip(data, sentence_results):
            for sent in result:
                if self.jaccard_similarity(sent.split(), example["sentence"].split()) > 0.8:
                    yield 1, 1
                else:
                    yield 0, 1

    def process_document_retrieval(self, data, doc_results):
        preds = []
        labels = []
        for idx, (example, result) in enumerate(zip(data, doc_results)):
            for rank, (doc, score, title) in enumerate(sorted(result, key=lambda x: x[1], reverse=True)):
                preds.append({
                    "query": idx,
                    "q0": "q0",
                    "docid": title,
                    "rank": rank,
                    "score": score,
                    "system": "system"
                })
            labels.append({
                "query": idx,
                "q0": "q0",
                "docid": example["title"],
                "rel": 1
            })

        return [pd.DataFrame(preds).to_dict(orient="list")], [pd.DataFrame(labels).to_dict(orient="list")]

    def run(self, data, doc_results, sentence_results, verified_results):
        doc_preds, doc_labels = self.process_document_retrieval(data, doc_results)
        sentence_preds, sentence_labels = zip(*self.process_sentence_selection(data, sentence_results))

        result = collections.defaultdict(dict)
        print(len(verified_results))
        print(len(data))

        for step, preds, labels in zip(["document_retrieval", "sentence_selection", "table_verification"], [doc_preds, sentence_preds, verified_results], [doc_labels, sentence_labels, [x["label"] for x in data]]):
            for name, metric in self.step2name2metrics[step].items():
                print(step, name, metric.compute(predictions=preds, references=labels))
                result[step][name] =  metric.compute(predictions=preds, references=labels).items()
        print(result)
        return result
