import pyrootutils
from typing import List
from kilt.knowledge_source import KnowledgeSource
from blingfire import text_to_sentences
from typing import Optional
import jsonlines
import random
from kilt.knowledge_source import KnowledgeSource

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=True,
    cwd=True,
)


from tango import Step, JsonFormat, Format


@Step.register("totto_input")
class TottoInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "0011"

    def run(self, input_file, size=-1):
        with jsonlines.open(input_file) as reader:
            data = list(reader)
            random.seed(21)
            random.shuffle(data)
            data = data
            if size != -1:
                data = data[:size]

        for i in range(len(data)):
            data[i]["linearized_table"] = data[i]["subtable_metadata_str"]
            data[i]["title"] = data[i]["table_page_title"]
        return data

