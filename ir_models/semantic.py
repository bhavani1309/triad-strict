from sentence_transformers import SentenceTransformer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore


class SemanticModel:

    def __init__(self):
        print("Loading Sentence-BERT model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def compute_similarity(self, source_artifacts, target_artifacts):

        source_ids = list(source_artifacts.keys())
        target_ids = list(target_artifacts.keys())

        source_texts = [" ".join(source_artifacts[i]) for i in source_ids]
        target_texts = [" ".join(target_artifacts[i]) for i in target_ids]

        source_embeddings = self.model.encode(source_texts)
        target_embeddings = self.model.encode(target_texts)

        sim_matrix = cosine_similarity(source_embeddings, target_embeddings)

        return sim_matrix