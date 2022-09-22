import ujson as json
import torch
import pyrootutils
import hydra
import jsonlines
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from read.dpr.searcher import DPRSearcher

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.report import Report
from read.utils.io import write_open


class Options(DPROptions):
    def __init__(self):
        super().__init__()
        self.dpr_training_data = ''
        self.kilt_training_data = ''
        self.output_file = ''
        self.num_negatives = 1
        self.batch_size = 16
        self.n_docs = 20
        # also need one of qry_encoder_path OR rag_model_path
        self.__required_args__ = ['dpr_training_data', 'kilt_training_data',
                                  'corpus_endpoint', 'output_file']



opts = Options().fill_from_args()



def find_hard_negative(answers, positive, docs):
    # try to find a retrieved document different from the positive that does not contain any answers
    negatives = []
    for title, text in zip(docs['title'], docs['text']):
        content = ' ' + title + ' ' + text + ' '
        if content == positive:
            continue
        answer_bearing = False
        for ans in answers:
            if content.find(ans) != -1:
                answer_bearing = True
                break
        if not answer_bearing:
            negatives.append((title, text))
            if len(negatives) >= opts.num_negatives:
                break
    return negatives


def process_batch(searcher, inst_batch, cfg, out):
    docs_batch = searcher.batch_query([inst['query'] for inst in inst_batch], k=cfg.k)
    assert len(docs_batch) == len(inst_batch)
    for inst, docs in zip(inst_batch, docs_batch):
        pos_title, pos_text = inst['positive']['title'], inst['positive']['text']
        pos_content = ' ' + pos_title + ' ' + pos_text + ' '
        negatives = find_hard_negative(inst['answers'], pos_content, docs)
        if negatives:
            inst['negatives'] = [{'title': title, 'text': text} for title, text in negatives]
            inst['answers'] = list(inst['answers'])  # retain for reference
            out.write(json.dumps(inst)+'\n')
        else:
            # CONSIDER: show sample of excluded results
            pass


@hydra.main(version_base="1.2", config_path=root / "config" / "eval", config_name="dpr.yaml")
def main(cfg):
    id2inst = dict()
    with jsonlines.open(opts.dpr_training_data) as reader:
        for jobj in reader:
            id2inst[jobj['id']] = jobj

    missing_count = 0
    with jsonlines.open(cfg.positive_pids) as reader:
        for jobj in reader:
            inst_id = jobj['id']
            if inst_id not in id2inst:
                missing_count += 1
                continue
            # normalize the answers
            norm_ans = set()
            for output in jobj['output']:
                if 'answer' in output:
                    normed = output['answer']
                    if len(normed) > 0:
                        norm_ans.add(' ' + normed + ' ')
            id2inst[inst_id]['answers'] = norm_ans

    if missing_count > 0:
        print(f'{opts.dpr_training_data} is missing {missing_count} instances from {opts.kilt_training_data}')

    torch.set_grad_enabled(False)
    searcher = DPRSearcher(cfg.faiss_index, cfg.lucene_index, cfg.question_encoder)
    out = write_open(opts.output_file)
    report = Report()
    inst_batch = []
    for inst in id2inst.values():
        if report.is_time():
            print(f'{report.progress_str()}')
        inst_batch.append(inst)
        if len(inst_batch) == opts.batch_size:
            process_batch(searcher, inst_batch)
            inst_batch = []
    if len(inst_batch) > 0:
        process_batch(inst_batch)

out.close()
