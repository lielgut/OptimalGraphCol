import numpy as np


def greedy_graph_col(g, k=None):
    if k is not None and k <= 0:
        raise Exception('Invalid Input')

    n = g.shape[0]
    col = [-1] * n
    max_col = 0
    col[0] = 0

    for i in range(1, n):
        avail_cols = set(range(0, max_col + 1))
        for j in range(0, n):
            if g[i][j] == 1:
                avail_cols.discard(col[j])
        if len(avail_cols) != 0:
            col[i] = min(avail_cols)
        else:
            col[i] = max_col + 1
            max_col += 1
            if k is not None:
                if max_col > k:
                    return None
    if k is None:
        return max_col+1, tuple(col)
    return k, tuple(col)