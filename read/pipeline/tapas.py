from tango import Step, Format, JsonFormat
import pandas as pd
import json
from typing import Optional 

@Step.register("tapas::table_linearization")
class TableLinearization(Step):
    DETERMINISTIC: bool = True
    CACHEABLE: bool = True
    FORMAT: Format = JsonFormat()
    VERSION: Optional[str] = "007"

    def run(self, data):
        for item in data:
            table = pd.DataFrame(item["table"])
            cells = zip(*item["highlighted_cells"])
            cells = [list(x) for x in cells]
            sub_table = table.iloc[cells[0], cells[1]].reset_index().astype(str)
            sub_table = sub_table.dropna(axis=1, how='all')
            item["table"] = sub_table.to_dict(orient="records")
        return data