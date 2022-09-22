import regex as re

class CellSelector:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, query, cell_value, cell_column):
        for num_str in re.findall(r"\d+(.\d+)?", cell_value):
            if num_str in query:
                return True

        if cell_value.lower() in query.lower():
            return True

        return False