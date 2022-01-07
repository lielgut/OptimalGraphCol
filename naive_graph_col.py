import random
import numpy as np

class Graph():
    def __init__(self, g):
        self.n = g.shape[0]
        self.graph = g

    def isValidCol(self, v, col, c):
        for i in range(self.n):
            if self.graph[v][i] == 1 and col[i] == c:
                return False
        return True
      
    def graphColRec(self, m, col, v):
        if v == self.n:
            return True
        for c in range(1, m + 1):
            if self.isValidCol(v, col, c):
                col[v] = c
                if self.graphColRec(m, col, v + 1):
                    return True
                col[v] = 0
  
    def graphCol(self, k):
        col = [0] * self.n
        return col if (self.graphColRec(k, col, 0) != None) else None

def min_graph_col_naive(g):
    graph = Graph(g)
    n = g.shape[0]
    for k in range(1, n):
        col = graph.graphCol(k)
        if col != None:
            return k, col
    return n, list(range(n))