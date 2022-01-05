import numpy as np
import itertools
import random

# generate graph
def gen_graph(k):
    r = random.randint(0, 5)
    g = np.zeros((k,k), dtype=int)
    for i in range(k):
        for j in range(i + 1, k):
            g[i][j] = g[j][i] = 1 if random.randint(0, 6) > r else 0
    return g

# count unique colors in given coloring
count_colors = lambda col : len(np.unique(np.array(col)))

# find a legal coloring in list of colorings
def find_col(g, n, col_combs):
    for col in col_combs:
        is_legal_col = True
        for i in range(n):
            for j in range(i + 1, n):
                if col[i] == col[j] and g[i][j] == 1:
                    is_legal_col = False
                    break
            if not is_legal_col:
                break
        if is_legal_col:
            return count_colors(col), col
    return None

# input: undirected graph represented by adj matrix
# output: min num of colors required to color the graphs, a min-coloring of the graph
def graph_col_naive(g, k=None, exactly_k=False):
    n = g.shape[0]
    if k is None:
        # check if there are no edges
        if not 1 in g:
            return 1, np.zeros(n)
        # check if graph is a clique
        if not 0 in g:
            return n, np.array(list(range(n)))
        # get all k-cols for k = 1 ... n-1
        col_combs = list(itertools.product(list(range(n - 1)), repeat=n))
        # sort by number of colors used
        col_combs.sort(key = count_colors)
        # remove all 1-cols
        col_combs = col_combs[n-1:]
        # find a minimal legal coloring of the graph
        col = find_col(g, n, col_combs)
        # if a coloring was found return it, else return a n-coloring
        return col if (col is not None) else (n, np.array(list(range(n))))

    else:
        if k > 0:
            col_combs = list(itertools.product(list(range(k)), repeat=n))
            if exactly_k:
                col_combs = list(filter(lambda col : count_colors(col) == k, col_combs))
            else:
                col_combs.sort(key = count_colors)
            return find_col(g, n, col_combs)
        else:
            raise Exception('Invalid Input')