{
    trainer(name, data_path, lr=1e-5) ::  {
        ["input_data_" + name]: {
            type: "infotab::input_from_totto",
            input_dir: data_path,
            task: name
        },
        ["json_to_para_" + name]: {
            type: "infotab::json_to_para",
            dataset_dict: {
                type: "ref",
                ref: "input_data_" + name,
            },
            rand_perm: 1,
        },
        ["preprocess_" + name]: {
            type: "infotab::preprocess",
            single_sentence: 2,
            dataset_dict: {
                type: "ref",
                ref: "json_to_para_" + name,
            },
            tokenizer: "roberta-large"
        },
        ["train_" + name]: {
            type: "torch::train",
            model: {
                type: "infotab::model",
                model_name_or_path: "roberta-large",
            },
            training_engine: {
                optimizer: {
                    type: "torch::AdamW",
                    lr: lr,
              },
            },
            train_epochs: 10,
            dataset_dict: {
                "type": "ref",
                "ref": "preprocess_" + name,
            },
            train_dataloader: {
                batch_size: 8,
                shuffle: false,
                collate_fn: {
                    type: "transformers::DataCollatorWithPadding",
                    tokenizer: {
                        pretrained_model_name_or_path: "roberta-large",  
                    },
                },
            },
            callbacks: [{
                    type: "train::classification_score_callback"
            }],
            validation_split: "dev",
            validation_dataloader: {
                batch_size: 8,
                shuffle: false,
            },
            device_count: 1,
            checkpoint_every: 10000,
        },
        ["eval_" + name]: {
            type: "torch::eval",
            model: {
                type: "ref",
                ref: "train_" + name,
            },
            dataset_dict: {
                "type": "ref",
                "ref": "preprocess_" + name,
            },
            dataloader: {
                batch_size: 2,
                shuffle: false
            },
            callbacks: [
                {
                    type: "eval::classification_score_callback"
                },
            ],
            auto_aggregate_metrics: false,
            test_split: "dev",
        },
    }
}