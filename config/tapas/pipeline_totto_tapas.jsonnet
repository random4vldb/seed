local lib = import '../seed/trainer.libsonnet';

local sent_selection_train = {
    read_data_sent_selection: {
        type: "tapas::input_data",
        tokenizer: "google/tapas-base",
        train_file: "temp/seed/sent_selection/data/train.jsonl",
        dev_file: "temp/seed/sent_selection/data/dev.jsonl",
        task: "sent_selection"
    },
    train_sent_selection: {
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
            "ref": "read_data_sent_selection",
        },
        train_dataloader: {
            batch_size: 2,
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
            batch_size: 2,
            shuffle: false
        },
        device_count: 8
    },
    eval_sent_selection: {
        type: "eval::classification",
        model: {
            type: "ref",
            ref: "train_sent_selection",
        },
        dataset_dict: {
            "type": "ref",
            "ref": "read_data_sent_selection",
        },
        test_split: "validation",
        batch_size: 1
    },
};

local verification_train = {
    read_data_verification: {
        type: "tapas::input_data",
        tokenizer: "google/tapas-base",
        task: "verification",
        train_file: "data/totto2/triplets/train.jsonl",
        dev_file: "data/totto2/triplets/dev.jsonl"
    },
    train_verification: {
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
            "ref": "read_data_verification",
        },
        train_dataloader: {
            batch_size: 2,
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
            batch_size: 2,
            shuffle: false
        },
    },
    eval_verification: {
        type: "eval::classification",
        model: {
            type: "ref",
            ref: "train_verification",
        },
        dataset_dict: {
            "type": "ref",
            "ref": "read_data_verification",
        },
        test_split: "validation",
        batch_size: 2
    },
};

{
    steps: sent_selection_train + verification_train + {
        data_input: {
            type: "pipeline::input_totto",
            input_file: "data/totto2/augmented/dev.jsonl",
            size: -1
        },
        table_linearization: {
            type: "tapas::table_linearization",
            data: {
                type: "ref",
                ref: "data_input",
            },
        },
        document_retrieval: {
            type: "pipeline::document_retrieval",
            searcher: "hybrid",
            faiss_index: "pyserini_faiss_full",
            lucene_index: "temp/pyserini_index",
            qry_encoder: "temp/dpr/models/qry_encoder_dpr",
            ctx_encoder: "temp/dpr/models/ctx_encoder_dpr",
            data: {
                type: "ref",
                ref: "data_input"
            },
            batch_size: 16

        },
        sentence_selection: {
            type: "pipeline::sentence_selection",
            model: {
                type: "ref",
                ref: "train_sent_selection"
            },
            tokenizer: "google/tapas-base",
            doc_results: {
                type: "ref",
                ref: "document_retrieval",
            },
            data: {
                type: "ref",
                ref: "table_linearization",
            },
            batch_size: 2
        },
        table_verification: {
            type: "pipeline::table_verification",
            model: {
                type: "ref",
                ref: "train_verification"
            },
            tokenizer: "google/tapas-base",
            sentence_results: {
                type: "ref",
                ref: "sentence_selection",
            },
            data: {
                type: "ref",
                ref: "table_linearization",
            },
            batch_size: 2
        },
        evaluation: {
            type: "pipeline::evaluation",
            data: {
                type: "ref",
                ref: "data_input"
            },
            sentence_results: {
                type: "ref",
                ref: "sentence_selection",
            },
            verified_results: {
                type: "ref",
                ref: "table_verification",
            },
            doc_results: {
                type: "ref",
                ref: "document_retrieval",
            },
        },
    },
}