{
    "steps": {
        "train_sent_selection": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "seed_sent_selection",
                "model_name_or_path": "facebook/bart-large"
            },
            "trainer": {
                "type": "default",
                "max_epochs": 5,
                "log_every_n_steps": 3,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {"type": "pytorch_lightning::CSVLogger"},
                ],
                "accelerator": "gpu",
                "profiler": {
                    "type": "pytorch_lightning::SimpleProfiler",
                },
            },
            "datamodule":{
                "type": "seed_sent_selection_data",
                "tokenizer": "facebook/bart-large",
                "dataset_name_or_path": "temp/seed/sent_selection/data"
            }
        }
    }
}