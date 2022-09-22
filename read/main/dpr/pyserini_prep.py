import ujson as json
import re
import os
import hydra
import pyrootutils
from pathlib import Path

root = pyrootutils.setup_root(search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)

from read.utils.io import read_open, write_open


@hydra.main(version_base="1.2", config_path=root / "config" / "dpr", config_name="dpr_pyserini_prep.yaml")
def main(cfg):
    # pid, title, text -> id, contents
    outfiles = [write_open(os.path.join(cfg.output, f'{j}.json')) for j in range(cfg.file_count)]

    input_dir = Path(cfg.input)
    for file in input_dir.iterdir():
        for line_ndx, line in enumerate(read_open(file).readlines()):
            jobj = json.loads(line)
            f = outfiles[line_ndx % len(outfiles)]
            f.write(json.dumps({'id': jobj['pid'],
                                'title': jobj["title"],
                                'contents': jobj['text']})+'\n')
    for of in outfiles:
        of.close()


if __name__ == '__main__':
    main()