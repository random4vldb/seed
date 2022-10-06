from tango import Step
from pathlib import Path
import jsonlines


@Step.register("seed_correction_generate")
class SeedCorrectionGenerate(Step):
    def run(self, input_path):
        for file in Path(input_path).glob("*.jsonl"):
            with jsonlines.open(file, "r") as reader:
                for obj in reader:
                    yield obj