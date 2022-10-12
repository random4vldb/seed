local lib = import 'infotab_train.libsonnet';

local sent_selection_train = lib.trainer("temp/infotab/sent_selection/data/");
local verification_train = lib.trainer("data/totto2/triplets/");

{
    steps: sent_selection_train + verification_train + {
        data_input: {
            type: "pipeline::input_totto",
            input_file: "data/totto2/augmented/dev.jsonl",
            size: 100
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
                ref: "data_input"
            },
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