from tango import Step, Format
from tango.integrations.torch import TorchFormat
from typing import Optional
import jsonlines
import json
from tango.integrations.pytorch_lightning import LightningModule


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


@Step.register("pytorch_lightning::convert")
class PLConvert(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: Optional[bool] = True
    FORMAT: Format = TorchFormat()

    def run(self, model: LightningModule, state_dict):
        model.load_state_dict(state_dict["state_dict"])
        return model