from rank_bm25 import BM25Okapi # type: ignore
import numpy as np # type: ignore


class BM25Model:

    def __init__(self):
        self.source_ids = []
        self.target_ids = []
        self.target_corpus = []
        self.bm25 = None

    def build_dual_corpus(self, source_artifacts, target_artifacts):

        self.source_ids = list(source_artifacts.keys())
        self.target_ids = list(target_artifacts.keys())

        self.target_corpus = [
            target_artifacts[i] for i in self.target_ids
        ]

        self.bm25 = BM25Okapi(self.target_corpus)

        self.source_queries = [
            source_artifacts[i] for i in self.source_ids
        ]

    def similarity_dual(self):

        sim_matrix = []

        for query_tokens in self.source_queries:

            scores = self.bm25.get_scores(query_tokens)

            sim_matrix.append(scores)

        return np.array(sim_matrix)

    def build_single_corpus(self, artifacts):

        self.ids = list(artifacts.keys())
        corpus = [artifacts[i] for i in self.ids]

        self.bm25 = BM25Okapi(corpus)

        self.corpus = corpus

    def similarity_single(self):

        sim_matrix = []

        for tokens in self.corpus:

            scores = self.bm25.get_scores(tokens)
            sim_matrix.append(scores)

        return np.array(sim_matrix)