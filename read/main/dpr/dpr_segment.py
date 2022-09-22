import itertools
import ujson as json
import os
import re
import sys
from pathlib import Path


import hydra
import pyrootutils
from loguru import logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import write_open


def find_para_range(cfg, word_counts, start_para, paragraphs):
    # find the end_para that fills up our passage
    end_para = start_para + 1
    while (
        end_para < len(word_counts)
        and sum(word_counts[start_para:end_para]) < cfg.max_passage_words
    ):
        end_para += 1
    if (
        sum(word_counts[start_para:end_para]) > cfg.max_passage_words
        and sum(word_counts[start_para : end_para - 1]) >= cfg.min_passage_words
    ):
        # don't need the last paragraph
        end_para -= 1
    if end_para == len(paragraphs):
        # include some earlier paragraphs
        while (
            start_para > 0
            and sum(word_counts[start_para:end_para]) < cfg.min_passage_words
        ):
            start_para -= 1
    return start_para, end_para


def write(cfg, doc_id, title, paragraphs, word_counts, start_para):
    """
    write a passage starting at start_para
    :param doc_id:
    :param title:
    :param paragraphs:
    :param word_counts:
    :param start_para:
    :return: the next start_para for a passage
    """
    assert len(paragraphs) == len(word_counts)
    assert 0 <= start_para < len(paragraphs)

    orig_start_para = start_para
    start_para, end_para = find_para_range(cfg, word_counts, start_para, paragraphs)
    full_end_para = end_para

    # enough words or all the words
    words = list(itertools.chain(*paragraphs[start_para:end_para]))
    assert len(words) >= cfg.min_passage_words or (
        start_para == 0 and end_para == len(paragraphs)
    )

    if cfg.min_passage_words <= len(words) <= cfg.max_passage_words:
        # our paragraph boundaries work out
        passage_id = f"{doc_id}::[{start_para},{end_para-1}]"
    elif len(words) > cfg.max_passage_words:
        # we need to truncate some of the first or last paragraph
        if start_para == orig_start_para:
            # chop from the end
            assert (
                start_para == end_para + 1
                or sum(word_counts[start_para : end_para - 1]) < cfg.min_passage_words
            )
            words = words[: cfg.max_passage_words]
            passage_id = f"{doc_id}::[{start_para},{end_para-1})"
            full_end_para = end_para - 1
        else:
            # chop from the begining
            assert (
                start_para == end_para + 1
                or sum(word_counts[start_para + 1 : end_para]) < cfg.min_passage_words
            )
            words = words[-cfg.max_passage_words :]
            passage_id = f"{doc_id}::({start_para},{end_para-1}]"
    else:
        # the document is too short, we take it all as a single too-short passage
        assert len(words) < cfg.min_passage_words and end_para - start_para == len(
            paragraphs
        )
        passage_id = f"{doc_id}::[{start_para},{end_para-1}]"
    assert len(words) <= cfg.max_passage_words
    text = " ".join(words)
    obj = {"pid": passage_id, "title": title, "text": text}

    return obj, passage_id, max(orig_start_para + 1, full_end_para)


_WHITESPACE = re.compile(r"\s+")


def clean_text(paragraph: str):
    # handle:
    #  Section::::
    #  BULLET::::-
    return _WHITESPACE.sub(" ", paragraph.replace("::::", ": ")).strip()


@hydra.main(
    version_base="1.2", config_path=root / "config", config_name="dpr_segment.yaml"
)
def main(cfg):
    out_files = [
        write_open(os.path.join(cfg.output_dir, f"{i}.jsonl.gz"))
        for i in range(cfg.num_output_files)
    ]
    passage_id_file = write_open(cfg.passage_ids)
    # passage id format is: doc_id::[start_para,end_para]
    # if the interval is doc_id::[start_para,end_para) then some (but not all) of end_para is included
    # if the interval is doc_id::(start_para,end_para] then some (but not all) of start_para is included
    passage_count = 0
    too_short_document_count = 0
    too_long_paragraph_count = 0
    total_paragraphs = 0
    kilt_corpus_path = Path(cfg.kilt_corpus)
    line_count = 0
    if kilt_corpus_path.is_dir():
        files = kilt_corpus_path.glob("*.json")
    else:
        files = [kilt_corpus_path]
    for file in files:
        with open(file, "r") as reader:
            for line in reader:
                line_count += 1
                jobj = json.loads(line)
                doc_id = jobj["wikipedia_id"]
                title = jobj["wikipedia_title"]
                paragraphs = [clean_text(p).split(" ") for p in jobj["text"]]
                word_counts = [len(p) for p in paragraphs]
                total_paragraphs += len(paragraphs)
                too_long_paragraph_count += sum(
                    [wc > cfg.max_passage_words for wc in word_counts]
                )
                too_short_document_count += (
                    1 if sum(word_counts) < cfg.min_passage_words else 0
                )
                start_para = 0
                while start_para < len(paragraphs):
                    obj, passage_id, start_para = write(
                        cfg, doc_id, title, paragraphs, word_counts, start_para
                    )
                    out_files[passage_count % len(out_files)].write(
                        json.dumps(obj) + "\n"
                    )
                    passage_id_file.write(f"{passage_id}\n")
                    passage_count += 1
                if line_count % 10000 == 0:
                    logger.info(f"Processed {line_count} lines, at file {file}")
                    logger.info(
                        f"{total_paragraphs} paragraphs, too long {too_long_paragraph_count}, docs too short {too_short_document_count}"
                    )
    logger.info(
        f"{total_paragraphs} paragraphs, too long {too_long_paragraph_count}, docs too short {too_short_document_count}"
    )
    for out_file in out_files:
        out_file.close()
    passage_id_file.close()


if __name__ == "__main__":
    main()
