from transformers import AutoTokenizer, AutoModel # type: ignore
import torch # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
import numpy as np #type:ignore


class CodeBERTModel:

    def __init__(self):
        print("Loading CodeBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.model = AutoModel.from_pretrained("microsoft/codebert-base")

    def encode(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding.numpy()

    def compute_similarity(self, source_artifacts, target_artifacts):

        source_ids = list(source_artifacts.keys())
        target_ids = list(target_artifacts.keys())

        source_texts = [" ".join(source_artifacts[i]) for i in source_ids]
        target_texts = [" ".join(target_artifacts[i]) for i in target_ids]

        source_embeddings = []
        target_embeddings = []

        for t in source_texts:
            source_embeddings.append(self.encode(t)[0])

        for t in target_texts:
            target_embeddings.append(self.encode(t)[0])

        source_embeddings = np.array(source_embeddings)
        target_embeddings = np.array(target_embeddings)

        sim_matrix = cosine_similarity(source_embeddings, target_embeddings)

        return sim_matrix