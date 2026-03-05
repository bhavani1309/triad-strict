import os
from preprocessing.biterm_generator import generate_biterms

import re

def split_camel_case(identifier):
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', identifier)
def read_recursive_files(base_path):
    """
    Recursively read all .txt files under a directory.
    Returns dictionary: {artifact_id: concatenated_text}
    """

    artifacts = {}

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".txt"):
                artifact_id = file.replace(".txt", "")
                file_path = os.path.join(root, file)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().lower()

                if artifact_id not in artifacts:
                    artifacts[artifact_id] = ""

                artifacts[artifact_id] += " " + text

    return artifacts


# ---------------- REQUIREMENTS ----------------
def load_requirements_biterm(base_path):

    raw_artifacts = read_recursive_files(base_path)

    artifacts = {}

    for artifact_id, text in raw_artifacts.items():

        tokens = [
            t for t in text.split()
            if len(t) > 2
        ]

        biterms = generate_biterms(tokens)

        artifacts[artifact_id] = biterms

    return artifacts


# ---------------- CODE ----------------
def load_code_biterm(base_path):

    raw_artifacts = read_recursive_files(base_path)

    artifacts = {}

    for artifact_id, text in raw_artifacts.items():
        text=split_camel_case(text)
        tokens = [
            t.lower() for t in text.split()
            if len(t) > 2
        ]

        biterms = generate_biterms(tokens)

        # identifier-aware weighting
        combined = tokens + biterms + tokens + tokens

        # give extra weight to identifiers
        weighted = combined * 2

        artifacts[artifact_id] = weighted

    return artifacts