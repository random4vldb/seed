from tango import Step, Format, JsonFormat
from typing import Optional
from transformers import AutoTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator


<<<<<<< HEAD

=======
>>>>>>> refactor: reorganize pipeline file structures
Step.register("tapas_sent_selection")
class TapasSentSelection(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, model, tokenizer, data, doc_results) -> str:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        all_preds = []

        queries = [x['linearized_table'] for x in data]
        inputs = tokenizer(list(zip(queries, doc_results)), return_tensors="pt", padding=True, truncation=True)

        dataset = Dataset.from_dict(inputs)

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4)

        accelerator = Accelerator()

        model, tokenizer, dataloader = accelerator.prepare(model, tokenizer, dataloader)

        for batch in dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=1)

            all_preds.extend(preds.tolist())

        all_preds.extend(preds)
        return all_preds

