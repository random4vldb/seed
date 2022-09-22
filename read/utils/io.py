import os
import gzip
import bz2


def read_open(input_file, *, binary=False, errors=None):
    """
    Open text file for reading, assuming compression from extension
    :param input_file:
    :return:
    """
    if not isinstance(input_file, str):
        input_file = str(input_file)
    if binary:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rb")
        elif input_file.endswith(".bz2"):
            return bz2.open(input_file, "rb")
        else:
            return open(input_file, "rb")
    else:
        if input_file.endswith(".gz"):
            return gzip.open(input_file, "rt", encoding="utf-8", errors=errors)
        elif input_file.endswith(".bz2"):
            return bz2.open(input_file, "rt", encoding="utf-8", errors=errors)
        else:
            return open(input_file, "r", encoding="utf-8", errors=errors)


def write_open(output_file, *, mkdir=True, binary=False):
    """
    Open text file for writing, assuming compression from extension
    :param output_file:
    :param mkdir:
    :return:
    """
    if not isinstance(output_file, str):
        output_file = str(output_file)
    if mkdir:
        dir = os.path.split(output_file)[0]
        if dir:
            os.makedirs(dir, exist_ok=True)
    if binary:
        if output_file.endswith(".gz"):
            return gzip.open(output_file, "wb")
        elif output_file.endswith(".bz2"):
            return bz2.open(output_file, "wb")
        else:
            return open(output_file, "wb")
    else:
        if output_file.endswith(".gz"):
            return gzip.open(output_file, "wt", encoding="utf-8")
        elif output_file.endswith(".bz2"):
            return bz2.open(output_file, "wt", encoding="utf-8")
        else:
            return open(output_file, "w", encoding="utf-8")
