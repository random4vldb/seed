from tango import Step, Format
from typing import Optional
import jsonlines
import json


@Step.register("io:file_output")
class FileOutput(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = str

    def run(self, data, output_file):
        if output_file.endswith(".json"):
            json.dump(data, open(output_file, "w"))
        elif output_file.endswith(".jsonl"):
            with jsonlines.open(output_file, "w") as f:
                f.write_all(data)
        else:
            raise ValueError("Unknown file format")