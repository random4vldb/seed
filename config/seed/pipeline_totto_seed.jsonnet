local lib = import '../seed/trainer.libsonnet';

local sent_selection_train = {
    "train_sent_selection": {
        "type": "pytorch_lightning::train",
        "model": {
            "type": "seed::sent_selection_model",
            "model_name_or_path": "facebook/bart-large"
        },
        "trainer": lib.trainer(name="sent_selection") + {
            max_epochs: 10
        },
        "datamodule":{
            "type": "seed::sent_selection_data",
            "tokenizer": "facebook/bart-large",
            "dataset_name_or_path": "temp/seed/sent_selection/data",
            "train_batch_size": 16,
            "eval_batch_size": 16
        }
    },
    convert_sent_selection: {
        type: "pytorch_lightning::convert",
        model: {
            type: "seed::sent_selection_model",
            model_name_or_path: "facebook/bart-large"
        },
        state_dict: {
            type: "ref",
            ref: "train_sent_selection",
        },
    },
};

local verification_train = {
    "train_verification": {
        "type": "pytorch_lightning::train",
        "model": {
            "type": "seed::verification_model",
            "model_name_or_path": "facebook/bart-large"
        },
        "trainer": lib.trainer(name="verification") + {
            max_epochs: 10
        },
        "datamodule":{
            "type": "seed::verification_data",
            "tokenizer": "facebook/bart-large",
            "dataset_name_or_path": "data/totto2/triplets",
            "train_batch_size": 8,
            "eval_batch_size": 8
        }
    },
    convert_verification: {
        type: "pytorch_lightning::convert",
        model: {
            type: "seed::verification_model",
            model_name_or_path: "facebook/bart-large"
        },
        state_dict: {
            type: "ref",
            ref: "train_verification",
        },
    },

};

{
    steps: sent_selection_train + verification_train + {
        data_input: {
            type: "pipeline::input_totto",
            input_file: "data/totto2/augmented/dev.jsonl",
            size: -1
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
            batch_size: 64

        },
        sentence_selection: {
            type: "pipeline::sentence_selection",
            model: {
                type: "ref",
                ref: "convert_sent_selection"
            },
            tokenizer: "facebook/bart-large",
            doc_results: {
                type: "ref",
                ref: "document_retrieval",
            },
            data: {
                type: "ref",
                ref: "data_input"
            },
            batch_size: 16
        },
        table_verification: {
            type: "pipeline::table_verification",
            model: {
                type: "ref",
                ref: "convert_verification"
            },
            tokenizer: "facebook/bart-large",
            sentence_results: {
                type: "ref",
                ref: "sentence_selection",
            },
            data: {
                type: "ref",
                ref: "data_input"
            },
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