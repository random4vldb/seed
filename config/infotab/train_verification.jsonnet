local splits = [
    "train",
    "dev",
];

local split_steps =  {
        input_data: {
            type: "infotab_input_from_totto",
            input_dir: "data/totto2/triplets/"
        },
        json_to_para: {
            type: "infotab_json_to_para",
            dataset_dict: {
                type: "ref",
                ref: "input_data",
            },
            rand_perm: 1,
        },
        preprocess: {
            type: "infotab_preprocess",
            single_sentence: 2,
            dataset_dict: {
                type: "ref",
                ref: "json_to_para",
            },
            tokenizer: "roberta-large"
        },
};


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
            train_epochs: 10,
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
                    type: "classify_score_callback"
                },
            ],
            auto_aggregate_metrics: false,
            test_split: "dev",
        },
    },
}