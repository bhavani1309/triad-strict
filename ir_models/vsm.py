from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore


class VSMModel:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            lowercase=False,
            norm='l2',
            smooth_idf=True,
            
        )

        self.source_ids = []
        self.target_ids = []

    def build_dual_corpus(self, source_dict, target_dict):

        self.source_ids = list(source_dict.keys())
        self.target_ids = list(target_dict.keys())

        corpus = list(source_dict.values()) + list(target_dict.values())
        tfidf_matrix = self.vectorizer.fit_transform(corpus)

        self.source_matrix = tfidf_matrix[:len(self.source_ids)]
        self.target_matrix = tfidf_matrix[len(self.source_ids):]

    def similarity_dual(self):
        return cosine_similarity(self.source_matrix, self.target_matrix)

    def build_single_corpus(self, artifact_dict):

        corpus = list(artifact_dict.values())
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.matrix = tfidf_matrix

    def similarity_single(self):
        return cosine_similarity(self.matrix)