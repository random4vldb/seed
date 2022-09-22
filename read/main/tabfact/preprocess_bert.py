from __future__ import division
import random
import sys
import io
import os
import logging
import re
import pandas as pd
import ujson as json
from tqdm import tqdm
from collections import Counter, OrderedDict
import pyrootutils
import hydra
import jsonlines
from pathlib import Path
import traceback

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)



program = os.path.basename(sys.argv[0])
L = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
L.info("Running %s" % ' '.join(sys.argv))

entity_linking_pattern = re.compile('@#.*?;-*[0-9]+,(-*[0-9]+)@#')
fact_pattern = re.compile('@#(.*?);-*[0-9]+,-*[0-9]+@#')
unk_pattern = re.compile('@#([^#]+);-1,-1@#')
TSV_DELIM = "\t"
TBL_DELIM = " ; "


def join_unicode(delim, entries):
    #entries = [_.decode('utf8') for _ in entries]
    return delim.join(entries)


def parse_fact(fact):
    fact = re.sub(unk_pattern, '[UNK]', fact)
    chunks = re.split(fact_pattern, fact)
    output = ' '.join([x.strip() for x in chunks if len(x.strip()) > 0])
    return output


def process_file(input_file, preprocessed_file, shuffle=False):

    examples = []
    with jsonlines.open(input_file, "r") as reader:
        tables = list([pd.DataFrame(json.loads(jobj["table"])) for jobj in reader])
    dataset = json.load(open(preprocessed_file, "r"))
    
    print(len(tables), len(dataset))
    
    for idx, (fname, sample) in tqdm(enumerate(dataset.items())):
        try:
            table = tables[int(fname.split("_")[0])]
            # print("File name", fname)
            # facts: list of strings
            # print(dataset[fname])
            facts = sample[0]
            # labels: list of ints

            labels = sample[1]
            assert all([x in [0, 1] for x in labels])
            assert len(facts) == len(labels)

            # types: list of table column strings
            types = [str(x) for x in table.columns.values.tolist()]

            # columns: {type: list of cell phrases in this column}
            columns = OrderedDict()
            for i, j in sample[4]:
                if table.columns[j] not in columns:
                    columns[table.columns[j]] = [table.iloc[i, j]]
                else:
                    columns[table.columns[j]].append(table.iloc[i, j])

            # pack into one example
            example = {
                "csv": fname,
                "columns": columns,
                "facts": facts,
                "labels": labels,
                "highlighted_cells": sample[4]
            }
            examples.append(example)

        except Exception as e:
            traceback.print_exc()
            print("{} is misformated".format(fname))
            raise e

    if shuffle:
        random.shuffle(examples)

    print("{} samples in total".format(len(examples)))

    return examples


def convert_to_tsv(out_file, examples, dataset_type, meta, scan):

    L.info("Processing {} examples...".format(dataset_type))
    total = 0

    unk = 0
    len_total = 0
    empty_table = 0
    with io.open(out_file, 'w', encoding='utf-8') as fout:
        for example in tqdm(examples):
            assert len(example['facts']) == len(example['labels'])
            for fact, label in zip(example['facts'], example['labels']):
                # use entity linking info to retain relevant columns
                # useful_column_nums = [int(x) for x in re.findall(entity_linking_pattern, fact) if not x == '-1']
                useful_column_nums = dict.fromkeys([idx for idx in range(len(example["columns"]))])
                remaining_table = OrderedDict()
                for idx, (column_type, column_cells) in enumerate(example['columns'].items()):
                    if idx in useful_column_nums:
                        column_type = '_'.join(column_type.split())
                        remaining_table[column_type] = column_cells

                fact_clean = parse_fact(fact)
                if len(remaining_table) > 0:
                    table_cells, table_feats = [], []

                    len_total += 1
                    if scan == 'vertical':
                        for column_type, column_cells in remaining_table.items():
                            column_type = ' '.join(column_type.split('_'))
                            table_cells.extend([column_type, 'are :'])
                            this_column = []
                            for idx, c in enumerate(column_cells):
                                this_column.append("row {} is {}".format(idx + 1, c))
                            this_column = join_unicode(TBL_DELIM, this_column)
                            table_cells.append(this_column)
                            table_cells.append('.')
                            table_feats.append(column_type)
                    else:
                        # stupid but to reserve order
                        table_column_names, table_column_cells = [], []
                        for column_type, column_cells in remaining_table.items():
                            column_type = ' '.join(column_type.split('_'))
                            table_feats.append(column_type)
                            table_column_names.append(column_type)
                            table_column_cells.append(column_cells)
                        for idx, row in enumerate(zip(*table_column_cells)):
                            table_cells.append('row {} is :'.format(idx + 1))
                            this_row = []
                            for col, tk in zip(table_column_names, row):
                                this_row.append('{} is {}'.format(col, tk))
                            this_row = join_unicode(TBL_DELIM, this_row)
                            table_cells.append(this_row)
                            table_cells.append('.')

                    table_str = ' '.join(table_cells)
                    out_items = [example['csv'],
                                 str(len(table_feats)),
                                 ' '.join([str(x) for x in table_feats]),
                                 table_str,
                                 fact_clean,
                                 str(label)]

                    out_items = TSV_DELIM.join(out_items)
                    total += 1
                    fout.write(out_items + "\n")
                else:
                    print(example["columns"], useful_column_nums, example["highlighted_cells"])
                    if dataset_type != 'train':
                        table_feats = ['[UNK]']
                        table_cells = ['[UNK]']
                        table_str = ' '.join(table_cells)
                        out_items = [example['csv'],
                                     str(len(table_feats)),
                                     ' '.join([str(x) for x in table_feats]),
                                     table_str,
                                     fact_clean,
                                     str(label)]

                        out_items = TSV_DELIM.join(out_items)
                        fout.write(out_items + "\n")
                        total += 1
                    empty_table += 1
    print("Built {} instances of features in total, {}/{}={}% unseen column types, {} empty tables"
          .format(total, unk, len_total, "{0:.2f}".format(unk * 100 / len_total), empty_table))
    meta["{}_total".format(dataset_type)] = total

    return meta


def save(filename, obj, message=None, beautify=False):
    assert message is not None
    print("Saving {} ...".format(message))
    with io.open(filename, "a") as fh:
        if beautify:
            json.dump(obj, fh, sort_keys=True, indent=4)
        else:
            json.dump(obj, fh)


def mkdir_p(path1, path2=None):
    if path2 is not None:
        path1 = os.path.join(path1, path2)
    if not os.path.exists(path1):
        os.mkdir(path1)
    return path1


def count_types(dataset):
    type_cnt = []
    for example in dataset:
        for name in example['columns'].keys():
            type_cnt.append('_'.join(name.split()))
    return type_cnt


@hydra.main(
    version_base="1.2",
    config_path=root / "config" / "tabfact",
    config_name="preprocess_bert.yaml",
)
def main(cfg):
    data = process_file(cfg.input_file, cfg.preprocessed_file)
    meta = {}
    Path(cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
    meta = convert_to_tsv(cfg.output_file, data, 'train', meta, cfg.scan)
    save(cfg.output_file + '.meta', meta, message='meta')


if __name__ == "__main__":
    main()
