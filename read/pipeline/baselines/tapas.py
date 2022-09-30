from ..seed import SEEDPipeline
from transformers import TapasForSequenceClassification, TapasTokenizer
import torch
import pandas as pd
from loguru import logger


class TapasPipeline(SEEDPipeline):
    def __init__(self, cfg):
        self.sent_selector = TapasForSequenceClassification.from_pretrained(
            "google/tapas-base"
        )
        self.sent_selector.load_state_dict(torch.load(cfg.sent_selection.model))
        self.sent_tokenizer = TapasTokenizer.from_pretrained(
            cfg.sent_selection.tokenizer
        )

        self.verifier = TapasForSequenceClassification.from_pretrained(
            "google/tapas-base"
        )
        self.verifier_tokenizer = TapasTokenizer.from_pretrained(cfg.verifier.tokenizer)
        self.verifier.load_state_dict(torch.load(cfg.verifier.model))

        self.cfg = cfg

    def process_one_batch(self, inputs):
        print(inputs["input_ids"].unsqueeze(0).shape)
        outputs = self.sent_selector(
            torch.cat([x["input_ids"].unsqueeze(0) for x in inputs]),
            torch.cat([x["attention_mask"].unsqueeze(0) for x in inputs]),
            torch.cat([x["token_type_ids"].unsqueeze(0) for x in inputs]),
        )
        return outputs.logits.argmax(dim=1).detach().cpu().numpy().tolist()

    def predict(self, examples):
        tables = [pd.DataFrame(x["table"]) for x in examples]
        input_batch = []
        all_outputs = []
        for i, table in enumerate(tables):
            input_ = self.sent_tokenizer(
                table,
                [x["sentence"] for x in examples],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_batch.append(input_)
            if len(input_batch) == self.cfg.batch_size:
                outputs = self.process_one_batch(input_batch)
                input_batch = []

                all_outputs.extend(outputs)
                logger.info("Processed {} examples", len(all_outputs))

        result = [False] * len(examples)
        input_batch = []
        for i, output in enumerate(all_outputs):
            if output == 1:
                output = self.verifier(**input_batch)
                result[i] = (
                    output.logits.argmax(dim=1).detach().cpu().numpy().tolist()[0] == 1
                )

        return [
            {"id": examples[i]["id"], "label": result[i]} for i in range(len(examples))
        ]
