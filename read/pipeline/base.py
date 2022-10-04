from tango import Step, JsonFormat, Format
from typing import Optional
import jsonlines
import random

@Step.register("pipeline_input_data")
class PipelineInputData(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = JsonFormat()

    def run(self, input_path):
        with jsonlines.open(input_path) as reader:
            data = list(reader)
            random.seed(21)
            random.shuffle(data)
            data = data
        return data
