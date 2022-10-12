local splits = [
    "train",
    "dev",
];

local split_steps = std.foldl(
    function(x, split) x + {
        ["input_data_" + split]: {
            type: "infotab_input_data",
            file: "temp/infotab/sent_selection/data/" + split + ".jsonl",
        },
        ["json_to_para_" + split]: {
            type: "infotab_json_to_para",
            data: {
                type: "ref",
                ref: "input_data_" + split,
            },
            rand_perm: 1,
        },
        ["preprocess_" + split]: {
            type: "infotab_preprocess",
            single_sentence: 2,
            data: {
                type: "ref",
                ref: "json_to_para_" + split,
            },
            tokenizer: "roberta-large"
        },
    },
    splits,
    {},
);


{
    steps: split_steps + {
        train: {
            type: "torch::train",
            model: {
                type: "infotab",
                model_name_or_path: "roberta-large",
            },
            training_engine: {
                optimizer: {
                    type: "torch::AdamW",
                    lr: 1e-5,
              },
            },
            train_epochs: 20,
            dataset_dict: {
                "type": "ref",
                "ref": "preprocess",
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
            validation_split: "dev",
            validation_dataloader: {
                batch_size: 4,
                shuffle: false
            },
        },
        eval: {
            type: "torch::eval",
            model: {
                type: "ref",
                ref: "train",
            },
            dataset_dict: {
                "type": "ref",
                "ref": "preprocess",
            },
            dataloader: {
                batch_size: 4,
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
    },
}