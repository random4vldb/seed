from sentence_transformers import SentenceTransformer, util

class SentenceSelector:
    def __init__(self, model_name_or_path: str, threshold: float = 0.32) -> None:
        self.model = SentenceTransformer(model_name_or_path)
        self.threshold = threshold


    def __call__(self, queries, sentences, indices):
        enc1 = self.model.encode(queries)
        enc2 = self.model.encode(sentences)


        return util.pairwise_cos_sim(enc1, enc2) > self.threshold

