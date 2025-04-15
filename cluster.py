from sklearn.cluster import KMeans
from pyvrp._pyvrp import Route, Solution, ProblemData, Depot, Client
from pyvrp import Model
from pyvrp.stop import MaxRuntime, MaxIterations
from utils import devide_subproblems, merge_subproblems, improve_subproblems
import numpy as np
import time
import math
import pdb

seed = 0


# st = time.time()
# X = np.random.rand(100000, 128)
# kmeans = KMeans(n_clusters=500, random_state=0, n_init="auto", init="k-means++").fit(X)
# print(kmeans.labels_)
# ed = time.time()
# print(ed - st)
# # 50  5.648131608963013s
# # 500 20.26532530784607

if __name__ == '__main__':
    # fmt: off
    COORDS = [
        (456, 320),  # location 0 - the depot
        (228, 0),    # location 1
        (912, 0),    # location 2
        (0, 80),     # location 3
        (114, 80),   # location 4
        (570, 160),  # location 5
        (798, 160),  # location 6
        (342, 240),  # location 7
        (684, 240),  # location 8
        (570, 400),  # location 9
        (912, 400),  # location 10
        (114, 480),  # location 11
        (228, 480),  # location 12
        (342, 560),  # location 13
        (684, 560),  # location 14
        (0, 640),    # location 15
        (798, 640),  # location 16
    ]
    DEMANDS = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]
    # fmt: on
    # from pyvrp import Model

    m = Model()
    m.add_vehicle_type(4, capacity=15)
    depot = m.add_depot(x=COORDS[0][0], y=COORDS[0][1])
    clients = [
        m.add_client(x=COORDS[idx][0], y=COORDS[idx][1], delivery=DEMANDS[idx])
        for idx in range(1, len(COORDS))
    ]

    locations = [depot] + clients
    for frm in locations:
        for to in locations:
            distance = abs(frm.x - to.x) + abs(frm.y - to.y)  # Manhattan
            m.add_edge(frm, to, distance=distance)

    sol = m.solve(stop=MaxIterations(10)).best

    for i in range(10):
        cost = 0
        for route in sol.routes():
            for i in range(len(route) - 1):
                cost += abs(m._clients[route[i] - 1].x - m._clients[route[i + 1] - 1].x) + abs(m._clients[route[i] - 1].y - m._clients[route[i + 1] - 1].y)
            cost += abs(m._clients[route[0] - 1].x - depot.x) + abs(m._clients[route[0] - 1].y - depot.y)
            cost += abs(m._clients[route[-1] - 1].x - depot.x) + abs(m._clients[route[-1] - 1].y - depot.y)
        print(f'Iterations : {i} Cost : {cost}')
        subproblems = devide_subproblems(sol, 5, m)
        # pdb.set_trace()
        sols = improve_subproblems(subproblems)
        sol, m = merge_subproblems(subproblems, sols)
        # pdb.set_trace()


