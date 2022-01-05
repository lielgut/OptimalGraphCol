import numpy as np


def dsatur_graph_col(g):
    n = g.shape[0]
    uncolored_vertices_and_saturation = [[i, 0] for i in range(0, n)]
    col = [-1] * n
    max_col = 0

    for i in range(0, n):
        uncolored_vertices_and_saturation.sort(key=lambda x: (sum(g[x[0]]), x[1]), reverse=True)
        avail_cols = set(range(0, max_col + 1))
        cur_vertex = uncolored_vertices_and_saturation[0][0]
        del uncolored_vertices_and_saturation[0]

        for j in range(0, n):
            if g[cur_vertex][j] == 1:
                avail_cols.discard(col[j])
        if len(avail_cols) != 0:
            col[cur_vertex] = min(avail_cols)
        else:
            col[cur_vertex] = max_col + 1
            max_col += 1
        for uncolored_vertice, saturation in uncolored_vertices_and_saturation:
            if g[cur_vertex][uncolored_vertice] == 1 and col[uncolored_vertice] == -1:
                saturation += 1

    return max_col + 1, tuple(col)
