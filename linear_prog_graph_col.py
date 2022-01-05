from pulp import *

def linear_prog_min_graph_col(g):

    n = g.shape[0]
    coloring = [-1] * n
    vertices = range(n)
    colors = range(n)
    lp = LpProblem(name='Linear_Programming_Graph_Coloring_Problem', sense=LpMinimize)
    colors_of_vertices = LpVariable.dicts(name='vertex_and_color', indices=(vertices, colors), lowBound=0, upBound=1,
                                       cat=LpInteger)
    is_color_used = LpVariable.dicts(name='color', indices=colors, lowBound=0, upBound=1, cat=LpInteger)
    objective = lpSum(is_color_used)
    lp += objective

    for vertex in vertices:
        vertex_sum = 0
        for color in colors:
            vertex_sum += colors_of_vertices[vertex][color]
        lp += (vertex_sum == 1)

    for i in range(0, n):
        for j in range(0, n):
            if g[i][j] == 1:
                for color in colors:
                    lp += colors_of_vertices[i][color] + colors_of_vertices[j][color] <= is_color_used[color]

    lp.solve(PULP_CBC_CMD(msg=0))

    for vertex in vertices:
        for color in colors:
            if colors_of_vertices[vertex][color].value() == 1:
                coloring[vertex] = color

    return int(value(lp.objective)), tuple(coloring)
