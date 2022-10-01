{
    steps: {
        read_data: {
            type: "tapas_input_data",
            tokenizer: "google/tapas-large",
            train_file: "temp/seed/sent_selection/data/train.jsonl",
            dev_file: "temp/seed/sent_selection/data/dev.jsonl"
        },
        train: {
            type: "torch::train",
            model: {
                type: "transformers::AutoModelForSequenceClassification::from_pretrained",
                pretrained_model_name_or_path: "google/tapas-large",
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
                        "pretrained_model_name_or_path": "google/tapas-large",  
                    },
                },
            },
            validation_split: "validation",
            validation_dataloader: {
                batch_size: 1,
                shuffle: false
            },
        },
        eval: {
            
        },
    },

}