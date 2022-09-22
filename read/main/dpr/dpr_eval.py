from pathlib import Path
from typing import List
import pprint

import hydra
import jsonlines
import numpy as np
import pyrootutils
import torch
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import write_open
from read.metrics.retrieval import DocRetrievalEvaluator


def get_tokenizer_and_model(cfg):
    qry_encoder = DPRQuestionEncoder.from_pretrained(cfg.qry_encoder_path)
    qry_encoder.eval()
    ctx_encoder = DPRContextEncoder.from_pretrained(cfg.ctx_encoder_path)
    ctx_encoder.eval()
    qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base')
    ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
    return qry_tokenizer, qry_encoder, ctx_tokenizer, ctx_encoder


def ctx_embed(doc_batch: List[dict], ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> np.ndarray:
    documents = {"title": [doci['title'] for doci in doc_batch], 'text': [doci['text'] for doci in doc_batch]}
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    # FIXME: maybe attention mask here too
    with torch.no_grad():
        embeddings = ctx_encoder(input_ids.to(device=ctx_encoder.device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


def qry_embed(qry_batch: List[str], qry_encoder: DPRQuestionEncoder, qry_tokenizer: DPRQuestionEncoderTokenizerFast) -> np.ndarray:
    inputs = qry_tokenizer(qry_batch, truncation=True, padding="longest", return_tensors="pt")  # max_length=self.hypers.seq_len_q,
    with torch.no_grad():
        embeddings = qry_encoder(inputs['input_ids'].to(device=qry_encoder.device),
                                 inputs['attention_mask'].to(device=qry_encoder.device), return_dict=True).pooler_output
    return embeddings.detach().cpu().to(dtype=torch.float16).numpy()


@hydra.main(version_base="1.2", config_path=root / "config" / "eval", config_name="dpr.yaml")
def main(cfg):
    qry_tokenizer, qry_encoder, ctx_tokenizer, ctx_encoder = get_tokenizer_and_model(cfg)

    predictions = []

    if Path(cfg.output).exists():
        predictions = [x for x in jsonlines.open(cfg.output)]
    else:
        with jsonlines.open(cfg.initial_retrieval) as reader:
            for jobj in reader:
                inst_id = jobj['id']
                query = jobj['query']
                passages = jobj['passages']
                pid2passage = {p['pid']: p for p in passages}
                # positive_pids = inst_id2pos_pids[inst_id]
                # target_mask = [p['pid'] in positive_pids for p in passages]
                # TODO: do some batching
                ctx_vecs = [ctx_embed([passage], ctx_encoder, ctx_tokenizer).reshape(-1) for passage in passages]
                qry_vec = qry_embed([query], qry_encoder, qry_tokenizer).reshape(-1)
                # now do dot products, create scored_pids
                scored_pids = [(p['pid'], np.dot(qry_vec, ctx_veci)) for p, ctx_veci in zip(passages, ctx_vecs)]
                # produce re-ranked output
                scored_pids.sort(key=lambda x: x[1], reverse=True)
                jobj['passages'] = [pid2passage[pid] for pid, _ in scored_pids]
                jobj["scores"] = [float(score) for _, score in scored_pids]
                # wids = to_distinct_doc_ids([passage['pid'] for passage in jobj['passages']])
                # jobj['doc_ids'] = wids
                predictions.append(jobj)
    
    golds = []
    with jsonlines.open(cfg.data) as reader:
        for jobj in reader:
            golds.append(jobj)
    # now do evaluation

    retrieval_evaluator = DocRetrievalEvaluator(cfg.ks)
    result = retrieval_evaluator.evaluate(predictions, golds)

    Path(cfg.output).parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(cfg.output, "w") as writer:
        writer.write_all(predictions)
    pprint.pprint(result)

if __name__ == "__main__":
    main()
