{
    steps: {
        read_data: {
            type: "tapas_input_data",
            tokenizer: "google/tapas-base",
            task: "verification",
            train_file: "data/totto2/triplets/train.jsonl",
            dev_file: "data/totto2/triplets/dev.jsonl"
        },
        train: {
            type: "torch::train",
            model: {
                type: "transformers::AutoModelForSequenceClassification::from_pretrained",
                pretrained_model_name_or_path: "google/tapas-base",
            },
            training_engine: {
                optimizer: {
                    type: "torch::AdamW",
                    lr: 1e-5,
              },
            },
            train_epochs: 5,
            dataset_dict: {
                "type": "ref",
                "ref": "read_data",
            },
            train_dataloader: {
                batch_size: 4,
                shuffle: true,
                collate_fn: {
                    type: "transformers::DataCollatorWithPadding",
                    tokenizer: {
                        pretrained_model_name_or_path: "google/tapas-base",  
                    },
                },
            },
            validation_split: "validation",
            validation_dataloader: {
                batch_size: 4,
                shuffle: false
            },
        },
        eval: {
            type: "classification_score",
            model: {
                type: "ref",
                ref: "train",
            },
            dataset_dict: {
                "type": "ref",
                "ref": "read_data",
            },
            test_split: "validation",
            batch_size: 4
        },
    },

}