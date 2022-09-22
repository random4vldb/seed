import re

_NON_WORD = re.compile(r'\W+')


def text_normalize(text: str):
    return _NON_WORD.sub(' ', text).lower().strip()
