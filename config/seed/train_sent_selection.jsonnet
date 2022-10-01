local lib = import 'trainer.libsonnet';

{
    "steps": {
        "train_sent_selection": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "seed_sent_selection",
                "model_name_or_path": "facebook/bart-large"
            },
            "trainer": lib.trainer(name="sent_selection"),
            "datamodule":{
                "type": "seed_sent_selection_data",
                "tokenizer": "facebook/bart-large",
                "dataset_name_or_path": "temp/seed/sent_selection/data",
                "train_batch_size": 16,
                "eval_batch_size": 16
            }
        }
    }
}