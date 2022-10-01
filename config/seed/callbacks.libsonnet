{
    "trainer": {
                "type": "default",
                "max_epochs": 5,
                "log_every_n_steps": 3,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {"type": "pytorch_lightning::CSVLogger"},
                ],
                "callbacks": [
                    {
                        "type": "pytorch_lightning::ModelCheckpoint",
                        "monitor": "val_loss"
                    },
                    {
                        "type": "pytorch_lightning::EarlyStopping",
                        "monitor": "val_loss"
                    },
                    {
                        "type": "pytorch_lightning::RichProgressBar"
                    }
                ],
                "accelerator": "gpu",
                "profiler": {
                    "type": "pytorch_lightning::SimpleProfiler",
                },
                strategy: "ddp"
            }
}