from pulp import *


def linear_prog_min_graph_col(g):
    n = g.shape[0]
    coloring = [-1] * n
    vertices = range(n)
    # colors = range(1, n+1)
    colors = range(1, max([int(sum(g[i])) for i in range(0, n)]) + 2)
    lp = LpProblem(name='Linear_Programming_Graph_Coloring_Problem', sense=LpMinimize)
    colors_of_vertices = LpVariable.dicts(name='vertex_and_color', indices=(vertices, colors), cat=LpBinary)
    is_color_used = LpVariable.dicts(name='color', indices=colors, cat=LpBinary)
    objective = lpSum(is_color_used)
    lp += objective

    for vertex in vertices:
        vertex_sum = 0
        for color in colors:
            vertex_sum += colors_of_vertices[vertex][color]
        lp += (vertex_sum == 1)

    first = 0
    for i in range(0, n):
        for j in range(first, n):
            if g[i][j] == 1:
                for color in colors:
                    lp += colors_of_vertices[i][color] + colors_of_vertices[j][color] <= 1
        first += 1

    for vertex in vertices:
        for color in colors:
            lp += colors_of_vertices[vertex][color] <= is_color_used[color]

    lp.solve(PULP_CBC_CMD(msg=0))

    for vertex in vertices:
        for color in colors:
            if colors_of_vertices[vertex][color].value() == 1:
                coloring[vertex] = color

    return int(value(lp.objective)), coloring
