import numpy as np # type: ignore


class RawEnrichmentEngine:

    def __init__(self, t=3):
        self.t = t

    def enrich_requirements(self, req_dict, dd_dict, sim_si):

        enriched = {}

        req_ids = list(req_dict.keys())
        dd_ids = list(dd_dict.keys())

        for i, req_id in enumerate(req_ids):

            new_tokens = list(req_dict[req_id])

            sims = sim_si[i]
            top_indices = np.argsort(-sims)[:self.t]

            for idx in top_indices:
                if sims[idx] > 0:
                    dd_id = dd_ids[idx]
                    new_tokens.extend(dd_dict[dd_id])

            enriched[req_id] = new_tokens

        return enriched