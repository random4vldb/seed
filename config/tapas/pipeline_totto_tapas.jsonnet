local lib = import '../seed/trainer.libsonnet';

local sent_selection_train = {
    read_data: {
        type: "tapas_input_data",
        tokenizer: "google/tapas-base",
        train_file: "temp/seed/sent_selection/data/train.jsonl",
        dev_file: "temp/seed/sent_selection/data/dev.jsonl"
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
            batch_size: 16,
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
            batch_size: 16,
            shuffle: false
        },
        device_count: 8
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
        batch_size: 16
    },
};

local verification_train = {
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
};

local verification_train = {
    "train_verification": {
            "type": "pytorch_lightning::train",
            "model": {
                "type": "seed_verification",
                "model_name_or_path": "facebook/bart-large"
            },
            "trainer": lib.trainer(name="seed_sent_selection") + {
                max_epochs: 10
            },
            "datamodule":{
                "type": "seed_verification_data",
                "tokenizer": "facebook/bart-large",
                "dataset_name_or_path": "data/totto2/triplets",
                "train_batch_size": 16,
                "eval_batch_size": 16
            }
        }
};

{
    steps: sent_selection_train + verification_train + {
        data_input: {
            type: "totto_input",
            input_file: "data/totto2/augmented/dev.jsonl",
            size: 10
        },
        document_retrieval: {
            type: "seed_document_retrieval",
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
            type: "seed_sentence_selection",
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
                ref: "data_input"
            },
        },
        table_verification: {
            type: "seed_table_verification",
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
                ref: "data_input"
            },
        }
    },
}