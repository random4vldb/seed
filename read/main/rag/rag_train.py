import ujson as json
import random
import logging
import torch
import torch.nn.functional as F
import hydra
import jsonlines
from transformers import RagTokenizer, RagTokenForGeneration
from loguru import logger
import pyrootutils


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.dpr.searcher import DPRSearcher


class Options(KgiHypers):
    def __init__(self):
        super().__init__()
        # TODO: test positive pid training
        self.include_positive_pids = ''
        self.provenance_loss_weight = 1.0


hypers = Options().fill_from_args()


def retrieve(queries, tokenizer):
    features = tokenizer(queries, padding=True, truncation=True, return_tensors="pt", max_length=hypers.max_seq_length)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")["input_ids"]
    input_dict = prepare_seq2seq_batch(tokenizer, queries, return_tensors="pt",
                                       max_length=hypers.max_context_length)
    input_ids = input_dict['input_ids'].to(model.device)
    attention_mask = input_dict['attention_mask'].to(model.device)
    context_input_ids, context_attention_mask, doc_scores, docs = \
        rest_retriever.retrieve(input_ids, attention_mask, n_docs=hypers.n_docs)
    return context_input_ids.reshape(len(queries) * hypers.n_docs, -1), \
           context_attention_mask.reshape(len(queries) * hypers.n_docs, -1), \
           doc_scores.reshape(len(queries), hypers.n_docs), \
           input_ids, attention_mask, docs


def one_batch(queries, answers, id_batch, tokenizer):
    global batch_count
    context_input_ids, context_attention_mask, doc_scores, input_ids, attention_mask, docs = retrieve(queries)

    panswers = prefered_answers(answers, docs, prefer_extractive=hypers.prefer_extractive)
    labels = prepare_seq2seq_batch_labels(tokenizer, panswers, return_tensors="pt",
                                          max_target_length=hypers.max_target_length).to(optimizer.hypers.device)

    if inst_id2pos_pids:
        gold_pids = [inst_id2pos_pids[inst_id] for inst_id in id_batch]
        target_mask = [[pid in positive_pids for pid in qdocs['pid']] for qdocs, positive_pids in zip(docs, gold_pids)]
        target_mask = torch.tensor(target_mask, dtype=torch.bool).to(optimizer.hypers.device)
        assert target_mask.shape == doc_scores.shape
        provenance_nll = -(F.log_softmax(doc_scores, dim=0).view(-1)[target_mask.view(-1)].sum())
    else:
        provenance_nll = 0
    outputs = optimizer.model(labels=labels,
                              context_input_ids=context_input_ids, context_attention_mask=context_attention_mask,
                              doc_scores=doc_scores)
    if batch_count == 0:
        print(f'logits shape = {outputs.logits.shape}')
    batch_count += 1
    loss = outputs.loss.mean() + hypers.provenance_loss_weight * provenance_nll
    loss_history.note_loss(loss.item())
    optimizer.step_loss(loss,
                        retrieval_time=rest_retriever.retrieval_time/(batch_count * hypers.per_gpu_train_batch_size))



@hydra.main(
    version_base="1.2", config_path=root / "config" / "rag", config_name="rag_train.yaml"
)
def main(cfg):
    inst_id2pos_pids = dict()
    with jsonlines.open(cfg.include_positive_pids) as reader:
        for obj in reader:
            inst_id2pos_pids[obj['inst_id']] = obj['positive_pids']

    tokenizer = RagTokenizer.from_pretrained(cfg.model_name_or_path)
    model = RagTokenForGeneration.from_pretrained(cfg.model_name_or_path)

    model.train()
    # construct rest retriever after the model
    searcher = DPRSearcher(cfg)
    rand = random.Random(hypers.seed)
    if hypers.fold:
        fold_num, fold_count = [int(part.strip()) for part in hypers.fold.split('of')]
        assert fold_num <= fold_count
        assert fold_count >= 1
        rel_by_fold, count_by_fold = get_relations_by_fold(hypers.kilt_data, fold_count)
        print(f'instance distribution by fold = {count_by_fold/count_by_fold.sum()}')
    else:
        fold_num, fold_count = 1, 1
        rel_by_fold = []
    query_batch = []
    answer_batch = []
    id_batch = []
    while True:
        optimizer.model.train()
        dataset = read_lines(hypers.kilt_data, shuffled_files=rand)
        for line_ndx, line in enumerate(block_shuffle(dataset, rand=rand, block_size=100000)):
            if line_ndx % hypers.world_size != hypers.global_rank:
                continue
            inst = json.loads(line)
            if hypers.fold:
                relation = get_relation_from_inst(inst)
                if relation in rel_by_fold[fold_num-1]:
                    # we exclude one fold
                    continue
            input_text = inst['input']
            query_batch.append(input_text)
            answer_batch.append(kilt_answers(inst,
                                             normalize_train_answer=hypers.normalize_train_answer,
                                             prefer_extractive=hypers.prefer_extractive,
                                             no_leading_space=hypers.no_leading_space))

            id_batch.append(inst['id'])
            if len(query_batch) == hypers.per_gpu_train_batch_size * hypers.n_gpu:
                one_batch(query_batch, answer_batch, id_batch)
                if not optimizer.should_continue():
                    return
                query_batch = []
                answer_batch = []
                id_batch = []


train()
(optimizer.model.module if hasattr(optimizer.model, "module") else optimizer.model).save_pretrained(hypers.output_dir)
logger.info(f'loss_history = {loss_history.loss_history}')
hypers.cleanup_corpus_server()
