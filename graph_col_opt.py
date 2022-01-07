from naive_graph_col import min_graph_col_naive
from linear_prog_graph_col import linear_prog_min_graph_col
from graph_col_ml_predict import predict_col_min
from graph_col_ml_predict import get_ml_model
from greedy_graph_col import greedy_graph_col
from dsatur_graph_col import dsatur_graph_col
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

graphs = [gen_graph(10, randint(0,8)) for i in range(1000)]
time_naive, time_lp, time_ml, time_greedy, time_dsatur = list(), list(), list(), list(), list()
ml_mis, greedy_mis, dsatur_mis = 0, 0, 0
ml_model = get_ml_model()

for g in graphs:
    st = time()
    k_naive, col_naive = min_graph_col_naive(g)
    end = time()
    time_naive.append(end - st)

    st = time()
    k_lp, col_lp = linear_prog_min_graph_col(g)
    end = time()
    time_lp.append(end - st)

    if k_lp != k_naive:
        print("NOOOOOOOOOOOOOOO")
        print(f'k_lp = {k_lp}, k_naive = {k_naive}')
        print(g)

    st = time()
    k_ml = predict_col_min(g, ml_model)
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

    st = time()
    k_dsatur, col_dsatur = dsatur_graph_col(g)
    end = time()
    time_dsatur.append(end - st)
    if k_dsatur != k_naive:
        dsatur_mis = dsatur_mis + 1

print('AVG TIME - NAIVE: {:.2f} ms'.format(np.mean(np.array(time_naive)) * 1000.0))
print('AVG TIME - LP: {:.2f} ms'.format(np.mean(np.array(time_lp)) * 1000.0))
print('AVG TIME - ML: {:.2f} ms (mistake = {}%)'.format(np.mean(np.array(time_ml)) * 1000.0, int(100 * (ml_mis / len(graphs)))))
print('AVG TIME - GREEDY: {:.2f} ms (mistake = {}%)'.format(np.mean(np.array(time_greedy)) * 1000.0, int(100 * (greedy_mis / len(graphs)))))
print('AVG TIME - DSATUR: {:.2f} ms (mistake = {}%)'.format(np.mean(np.array(time_dsatur)) * 1000.0, int(100 * (dsatur_mis / len(graphs)))))