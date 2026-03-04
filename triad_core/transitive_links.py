import numpy as np # type: ignore


class TransitiveEngine:

    def __init__(self, base_t=3, base_m=0.5):
        self.base_t = base_t
        self.base_m = base_m


    def compute_outer_transitive_bonus(self, s_i_matrix, i_t_matrix):

        num_sources = s_i_matrix.shape[0]
        num_targets = i_t_matrix.shape[1]

        bonus_matrix = np.zeros((num_sources, num_targets))

        for s in range(num_sources):

            sims = s_i_matrix[s]

            max_sim = np.max(sims)

            if max_sim == 0:
                continue

            threshold = self.base_m * max_sim

            candidates = [
                i for i in range(len(sims))
                if sims[i] >= threshold
            ]

            candidates.sort(key=lambda i: sims[i], reverse=True)

            candidates = candidates[:self.base_t]

            for i in candidates:

                for t in range(num_targets):

                    bonus_matrix[s, t] += sims[i] * i_t_matrix[i, t]

        return bonus_matrix


    def compute_inner_transitive_bonus(
        self,
        s_s_matrix,
        s_i_matrix,
        i_i_matrix,
        i_t_matrix
    ):

        num_sources = s_i_matrix.shape[0]
        num_targets = i_t_matrix.shape[1]

        bonus_matrix = np.zeros((num_sources, num_targets))

        for s in range(num_sources):

            for s2 in range(num_sources):

                if s == s2:
                    continue

                ss_sim = s_s_matrix[s][s2]

                if ss_sim <= 0:
                    continue

                for i in range(len(s_i_matrix[s2])):

                    si_sim = s_i_matrix[s2][i]

                    if si_sim <= 0:
                        continue

                    for t in range(num_targets):

                        it_sim = i_t_matrix[i][t]

                        if it_sim <= 0:
                            continue

                        bonus_matrix[s, t] += ss_sim * si_sim * it_sim

            # Intermediate-to-intermediate propagation
            for i in range(len(s_i_matrix[s])):

                si_sim = s_i_matrix[s][i]

                if si_sim <= 0:
                    continue

                for i2 in range(len(i_i_matrix)):

                    if i == i2:
                        continue

                    ii_sim = i_i_matrix[i][i2]

                    if ii_sim <= 0:
                        continue

                    for t in range(num_targets):

                        it_sim = i_t_matrix[i2][t]

                        if it_sim <= 0:
                            continue

                        bonus_matrix[s, t] += si_sim * ii_sim * it_sim

        return bonus_matrix