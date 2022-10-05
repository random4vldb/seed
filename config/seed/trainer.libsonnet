{
    trainer(name="model", profiling=false) :: {
                "type": "default",
                "log_every_n_steps": 3,
                "logger": [
                    {"type": "pytorch_lightning::TensorBoardLogger"},
                    {"type": "pytorch_lightning::CSVLogger"},
                ],
                "callbacks": [
                    {
                        type: "pytorch_lightning::ModelCheckpoint",
                        monitor: "val_loss",
                        save_top_k: 1,
                        dirpath: "models",
                        filename: name + "-{epoch:02d}-{val_loss:.2f}"
                    },
                    {
                        "type": "pytorch_lightning::EarlyStopping",
                        "monitor": "val_loss"
                    },
                    {
                        "type": "pytorch_lightning::RichProgressBar"
                    }
                ],
                profiler: if profiling then {
                    "type": "pytorch_lightning::SimpleProfiler",
                } else null,
                accelerator: "gpu",
                strategy: "ddp_spawn"
            }
}
