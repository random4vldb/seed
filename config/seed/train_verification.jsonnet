local lib = import "trainer.libsonnet";

{
    "steps": {
        "train": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "seed_verification",
                "model_name_or_path": "facebook/bart-large"
            },
            "trainer": lib.trainer("seed_sent_selection") + {
                max_epochs: 10
            },
            "datamodule":{
                "type": "seed_verification_data",
                "tokenizer": "facebook/bart-large",
                "dataset_name_or_path": "data/totto2/triplets",
                "train_batch_size": 16,
                "eval_batch_size": 16
            }
        }
    }
}