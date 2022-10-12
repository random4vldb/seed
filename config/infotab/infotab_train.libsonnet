local splits = [
    "train",
    "dev",
];


{
    trainer(name, data_path) :: 
            local split_steps = std.foldl(
            function(x, split) x + {
                ["input_data_" + split]: {
                    type: "infotab_input_data",
                    file: data_path + split + ".jsonl",
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


        split_steps + {
            "train": {
                type: "infotab_train",
                model_name_or_path: "roberta-large",
                train_data: {
                    type: "ref",
                    ref: "preprocess_train",
                },
                dev_data: {
                    type: "ref",
                    ref: "preprocess_dev",
                },
                test_data: {
                    type: "ref",
                    ref: "preprocess_dev",
                },
                batch_size: 4,
                num_epochs: 10,
                output_dir: "temp/infotab/sent_selection/models",
            },
        },
}