{
    steps: {
        read_data: {
            type: "datasets::load",
            path: "json",
            data_files: {
                train: "temp/seed/sent_selection/data/train.jsonl",
                dev: "temp/seed/sent_selection/data/dev.jsonl"
            },
        },
        train: {
            type: "torch::train",
            model: {
                type: "transformers::AutoModelForSequenceClassification::from_pretrained",
                pretrained_model_name_or_path: "microsoft/tapex-large",
            },
            training_engine: {
                optimizer: {
                    type: "torch::AdamW",
                    lr: 1e-5,
              },
            },
            train_epochs: 10,
            dataset_dict: {
                "type": "ref",
                "ref": "read_data",
            },
            train_dataloader: {
                batch_size: 1,
                shuffle: true,
                collate_fn: {
                    type: "transformers::DataCollatorWithPadding",
                    tokenizer: {
                        "pretrained_model_name_or_path": "microsoft/tapex-large",  
                    },
                },
            },
            validation_split: "validation",
            validation_dataloader: {
                batch_size: 1,
                shuffle: false
            },
        },
    },

}