class ResultAnalyzer:
    def __init__(self, golds) -> None:
        self.report = []
        self.golds = golds
        self.document_preds = [False] * len(golds)
        self.sentence_preds = [False] * len(golds)
        self.current_index = 0

    def check_document(self, document, i):
        if document["title"] == self.golds[self.current_index + i]["title"]:
            self.document_preds[self.current_index + i] = True

    def check_sentence(self, sentence, i):
        if sentence == self.golds[self.current_index + i]["sentence"]:
            self.sentence_preds[self.current_index + i] = True

    def print(self, output_file):
        if output_file is not None:
            with open(output_file, "w") as f:
                for i, gold in enumerate(self.golds):
                    f.write(f"{gold['title']} | {gold['sentence']} | {gold['linearized_table']} | {self.document_preds[i]} | {self.sentence_preds[i]}\n")
        else:
            for i, gold in enumerate(self.golds):
                print(f"{gold['title']} | {gold['sentence']} | {gold['linearized_table']} | {self.document_preds[i]} | {self.sentence_preds[i]}")


