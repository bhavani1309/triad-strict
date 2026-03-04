def generate_biterms(tokens, window_size=5):
    """
    Generate sliding-window based biterms.
    Only combine tokens within a local window.
    """

    biterms = []

    n = len(tokens)

    for i in range(n):
        for j in range(i+1, min(i + window_size, n)):
            if tokens[i] != tokens[j]:
                sorted_pair = sorted([tokens[i], tokens[j]])
                biterms.append(sorted_pair[0] + "_" + sorted_pair[1])

    return biterms