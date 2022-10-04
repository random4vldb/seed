{
    steps: {
        data_input: {
            type: "pipeline_input_data",
            input_file: "data/totto2/augmented/dev.jsonl"
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
        },
        sentence_selection: {
            type: "seed_sentence_selection",
            model: "temp/seed/sent_selection/models/model.ckpt",
            tokenizer: "facebook/bart-large",
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
            model: "temp/seed/table_verification/models/model.ckpt",
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
        table_correction: {
            type: "seed_table_correction",
            model: "temp/seed/table_correction/models/model.ckpt",
            tokenizer: "facebook/bart-large",
            verification_results: {
                type: "ref",
                ref: "table_verification",
            },
            data: {
                type: "ref",
                ref: "data_input"
            },
        },
    },
}