from tango import Step
import torch


@Step.register("tapex_input_data")
class TapexInputData(Step):
    def run(self, train_file, dev_file):
        torch.manual_seed(1)
        train_df = pd.read_json(train_file, lines=True)
        dev_df = pd.read_json(dev_file, lines=True)

        train_dataset = TableDataset(train_df, tokenizer)
        dev_dataset = TableDataset(dev_df, tokenizer)
        return DatasetDict(
            {
                "train": train_dataset,
                "dev": dev_dataset,
            }
        )