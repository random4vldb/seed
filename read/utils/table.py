import copy
import pandas as pd
import json
import six


def linearize(
    table,
    hightlighted_cells=None,
    value_sep=" : ",
    row_sep=" ; ",
    includes_header=True,
    return_text=True,
):
    table = pd.DataFrame(table)
    table = table.applymap(lambda x: " , ".join(x) if isinstance(x, list) else x)
    values = []
    if hightlighted_cells is None:
        for i in range(len(table)):
            for j in range(len(table.columns)):
                if includes_header:
                    values.append(table.columns[j] + value_sep + table.iloc[i, j])
                else:
                    values.append(table.iloc[i, j])
    else:
        for i, j in hightlighted_cells:
            if includes_header:
                values.append(table.columns[j] + value_sep + table.iloc[i, j])
            else:
                values.append(table.iloc[i, j])
    if return_text:
        input = row_sep.join(values)
        return input
    return values


def linearize_tapex(
    table,
    highlighted_cells=None,
    value_sep=" : ",
    row_sep=" ; ",
    includes_header=True,
    return_text=True,
):
    table = pd.DataFrame(table)
    table = table.applymap(lambda x: ", ".join(x) if isinstance(x, list) else x)
    values = []
    header_str = ""
    row_str = ""
    if highlighted_cells is None:
        header_str = "col : " + " | ".join(table.columns)
        for i in range(len(table)):
            row_cell_values = []
            for j in range(len(table.columns)):
                row_cell_values.append(table.iloc[i, j])
            row_str = "row : " + " | ".join(row_cell_values)
        return header_str + " " + row_str
    else:
        col_indices = set([x[1] for x in highlighted_cells])
        row_indices = set([x[0] for x in highlighted_cells])
        header_str = "col : " + " | ".join([table.columns[i] for i in col_indices])
        for i in row_indices:
            row_cell_values = []
            for j in col_indices:
                if [i, j] in highlighted_cells:
                    row_cell_values.append(table.iloc[i, j])
                else:
                    row_cell_values.append("")
            row_str = f"row {i + 1}: " + " | ".join(row_cell_values)
        return header_str + " " + row_str



def infotab2totto(example):
    table_str = ""
    table_str += "<page_title> " + example["title"] + " </page_title> "
    table_str += "<table> "

    table = pd.DataFrame(json.loads(example["table"]))

    for idx, row in table.iterrows():
        for col in table.columns:
            item_str = "<cell> " + " ".join(row[col]) + " "

            item_str += "<col_header> " + col + " </col_header> "
        
            item_str += "</cell> "
            table_str += item_str

    table_str += "</table>"
    return table_str