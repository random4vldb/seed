import re
from flair.data import Sentence
from flair.models import SequenceTagger

tagger = SequenceTagger.load('ner')
sentence = Sentence("Minh")
tagger.predict(sentence)