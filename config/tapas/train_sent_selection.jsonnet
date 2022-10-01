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
                type: "transformers::AutoModel::from_pretrained",
                pretrained_model_name_or_path: "google/tapas-large",
            },
        },
    },

},