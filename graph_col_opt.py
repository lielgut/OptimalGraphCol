from naive_graph_col import min_graph_col_naive
from linear_prog_graph_col import linear_prog_min_graph_col
from graph_col_ml_predict import predict_col_min
from graph_col_ml_predict import get_ml_model
from greedy_graph_col import greedy_graph_col
from random import randint
from time import time
import numpy as np
import random

def gen_graph(k, r):
    g = np.zeros((k,k), dtype=int)
    for i in range(k):
        for j in range(i + 1, k):
            g[i][j] = g[j][i] = 0 if random.randint(0, 9) > r else 1
    return g

graphs = [gen_graph(10, randint(0,9)) for i in range(100)]
time_naive, time_lin_prog, time_ml, time_greedy = list(), list(), list(), list()
ml_mis, greedy_mis = 0, 0
ml_model = get_ml_model()

for g in graphs:
    st = time()
    k_naive, col_naive = min_graph_col_naive(g)
    end = time()
    time_naive.append(end - st)

    st = time()
    k_lin_prog, col_lin_prog = linear_prog_min_graph_col(g)
    end = time()
    time_lin_prog.append(end - st)

    if (k_naive != k_lin_prog):
        print(f'LIN_PROG DIFF FROM NAIVE!!! (k_naive = {k_naive}, k_lin_prog = {k_lin_prog})\n')
        print('graph:')
        print(g)
        print('\ncol naive:')
        print(col_naive)
        print('col lin prog:')
        print(col_lin_prog)

    st = time()
    k_ml = predict_col_min(g, ml_model)
    # k_ml, col_ml = min_graph_col_naive(g, min_k= k_ml)
    end = time()
    time_ml.append(end - st)
    if k_ml != k_naive:
        ml_mis = ml_mis + 1

    st = time()
    k_greedy, col_greedy = greedy_graph_col(g)
    end = time()
    time_greedy.append(end - st)
    if k_greedy != k_naive:
        greedy_mis = greedy_mis + 1

print('AVG TIME - NAIVE: {:.2f} ms'.format(np.mean(np.array(time_naive)) * 1000.0))
print('AVG TIME - LP: {:.2f} ms'.format(np.mean(np.array(time_lin_prog)) * 1000.0))
print('AVG TIME - ML: {:.2f} ms (mistake = {}%)'.format(np.mean(np.array(time_ml)) * 1000.0, int(100 * (ml_mis / len(graphs)))))
print('AVG TIME - GREEDY: {:.2f} ms (mistake = {}%)'.format(np.mean(np.array(time_greedy)) * 1000.0, int(100 * (greedy_mis / len(graphs)))))