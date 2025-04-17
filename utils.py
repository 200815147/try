from sklearn.cluster import KMeans
from pyvrp._pyvrp import Route, Solution, ProblemData, Depot, Client
from pyvrp import Model
from pyvrp.stop import MaxRuntime, MaxIterations
from sklearn.neighbors import KDTree
import numpy as np
from pyvrp.plotting import (
    plot_coordinates,
    plot_instance,
    plot_result,
    plot_route_schedule,
)
import pdb
import torch
import os
import random
from tqdm import tqdm
import time


def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed) 

def dimacs_round(x: float) -> int:
    return int(10 * x) // 10

def devide_subproblems(sol: Solution, subproblems_size, data: Model) -> list[Model]:
    n = sol.num_clients()
    m = (n + subproblems_size - 1) // subproblems_size
    print(f'Num of subproblems: {m}')
    routes = sol.routes()
    clients = data._clients
    depot = data._depots[0]
    capacity = data._vehicle_types[0].capacity[0]
    X = np.array([route.centroid() for route in routes]) # Depot?
    kmeans = KMeans(n_clusters=m, random_state=0, n_init="auto", init="k-means++").fit(X)
    subproblems = [Model() for _ in range(m)]
    for i in range(sol.num_routes()):
        idx = kmeans.labels_[i]
        for j in range(len(routes[i])):
            subproblems[idx]._clients.append(clients[routes[i][j] - 1])
    for i in range(m):
        subproblems[i]._depots.append(depot)
        subproblems[i].add_vehicle_type(len(subproblems[i]._clients), capacity=capacity)
        # If not specified, the number of vehicles is n
        locations = subproblems[i]._depots + subproblems[i]._clients
        for frm in locations:
            for to in locations:
                distance = dimacs_round(np.sqrt((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2))
                subproblems[i].add_edge(frm, to, distance=distance, duration=distance)
    return subproblems

def improve_subproblems(subproblems: list[Model]) -> list[Solution]:
    return [subproblem.solve(stop=MaxIterations(500), display=False).best for subproblem in subproblems]

def merge_subproblems(subproblems: list[Model], sols: list[Solution | list[int]]) -> tuple[Solution, Model]:
    data = Model()
    routes = []
    data._depots = subproblems[0]._depots
    capacity = subproblems[0]._vehicle_types[0].capacity[0]
    offset = 0
    for i in range(len(subproblems)):
        if isinstance(sols[i], Route):
            tmp_route = [[x + offset for x in list(route)] for route in sols[i].routes()]
        else:
            tmp_route = [[x + offset for x in list(route)] for route in sols[i]]
        routes.extend(tmp_route)
        data._clients.extend(subproblems[i]._clients)
        # Merge edges is not needed
        offset += len(subproblems[i]._clients)
    data.add_vehicle_type(len(data._clients), capacity=capacity)
    x = np.array([data._depots[0].x] + [c.x for c in data._clients])
    y = np.array([data._depots[0].y] + [c.y for c in data._clients])
    x1 = x[:, np.newaxis]
    y1 = y[:, np.newaxis]
    matrix = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    matrix = (matrix * 10).astype(int) // 10
    problemdata = ProblemData(
            data._clients,
            data._depots,
            data.vehicle_types,
            [matrix],
            [matrix],
            data._groups,
        )
    return Solution(data=problemdata, routes=routes), data

def create_problem(x, y, demand, capacity, early=None, late=None, service=None) -> Model:
    m = Model()
    m.add_vehicle_type(len(x), capacity=capacity)
    depot = m.add_depot(x=x[0], y=y[0])
    if early:
        clients = [
            m.add_client(x=x[idx], y=y[idx], delivery=demand[idx],
                         tw_early=early[idx], tw_late=late[idx], service_duration=service[idx])
            for idx in range(1, len(x))
        ]
    else:
        clients = [
            m.add_client(x=x[idx], y=y[idx], delivery=demand[idx])
            for idx in range(1, len(x))
        ]
    locations = [depot] + clients
    for frm in locations:
        for to in locations:
            distance = dimacs_round(np.sqrt((frm.x - to.x) ** 2 + (frm.y - to.y) ** 2))
            if early:
                m.add_edge(frm, to, distance=distance, duration=distance)
            else:
                m.add_edge(frm, to, distance=distance)
    return m

def read_vrptw_problem(file_path):
    capacity, n, depot_id = None, None, None
    stage = None
    with open(file_path, "r") as f: # 1 为 depot，2~n 为 customer
        lines = f.readlines()
        for line in lines:
            if line.startswith('CAPACITY'):
                capacity = int(line.strip().split(' ')[-1])
            elif line.startswith('DIMENSION'):
                n = int(line.strip().split(' ')[-1])
                x = [0 for _ in range(n)]
                y = [0 for _ in range(n)]
                demand = [0 for _ in range(n)]
                early = [0 for _ in range(n)]
                late = [0 for _ in range(n)]
                service = [0 for _ in range(n)]
            elif line.startswith('NODE_COORD_SECTION'):
                stage = 'read_nodes'
            elif line.startswith('DEMAND_SECTION'):
                stage = 'read_demand'
            elif line.startswith('TIME_WINDOW_SECTION'):
                stage = 'read_time_window'
            elif line.startswith('SERVICE_TIME_SECTION'):
                stage = 'read_service_time'
            elif line.startswith('DEPOT_SECTION'):
                stage = 'read_depot'
            elif stage == 'read_nodes':
                # pdb.set_trace()
                idx, xx, yy = line.strip().split(' ')
                idx = int(idx) - 1
                x[idx] = int(xx)
                y[idx] = int(yy)
            
            elif stage == 'read_demand':
                idx, dd = line.strip().split(' ')
                demand[int(idx) - 1] = int(dd)
            
            elif stage == 'read_time_window':
                idx, ee, ll = line.strip().split(' ')
                idx = int(idx) - 1
                early[idx] = int(ee)
                late[idx] = int(ll)
            
            elif stage == 'read_service_time':
                idx, ss = line.strip().split(' ')
                service[int(idx) - 1] = int(ss)
            
            elif stage == 'read_depot':
                depot_id = int(line) - 1
                stage = None
    return x, y, demand, capacity, early, late, service, depot_id, n

def read_sol(file_path):
    routes = []
    sol_cost, sol_time = None, None
    with open(file_path, "r") as f: # 1 为 depot，2~n 为 customer
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('Route'):
                routes.append([int(v) for v in line.split(' ')[2:]])
            elif line.startswith('Cost'):
                sol_cost = float(line.split(' ')[-1])
            elif line.startswith('Time'):
                sol_time = float(line.split(' ')[-1])
    # pdb.set_trace()
    return routes, sol_cost, sol_time

def print_routes(sol: Solution):
    routes = sol.routes()
    cnt = 0
    for route in routes:
        cnt += 1
        print(f'Route #{cnt}: ')
        for v in route:
            print(v, end=' ')
        print('')

def greedy_nearest(cost_matrix, demand, capacity):
    cost = 0
    cur = 0
    x = 0
    tour = []
    cnt = 1
    n = len(cost_matrix)
    vis = [0 for _ in range(n + 2)]
    while cnt < n or x > 0:
        next = 0
        for i in range(1, n):
            if vis[i] == 0 and cur + demand[i] <= capacity:
                if next == 0:
                    next = i
                elif cost_matrix[x][i] < cost_matrix[x][next]:
                    next = i
        if next > 0:
            cur += demand[i]
            cnt += 1
            vis[next] = 1
        else:
            cur = 0
        cost += cost_matrix[x][next]
        x = next
        tour.append(x)
    return tour, cost

def get_k_nearest(points, k):
    tree = KDTree(points)
    k_nearest = []
    for i in range(len(points)):
        dist, ind = tree.query([points[i]], k=k)
        k_nearest.append(ind[0])
    return k_nearest

def improve_one_subproblem(sol: Solution, data: Model, start_time: float=None, k: int=10, iters: int=500):
    n = sol.num_clients()
    clients = [None] + data._clients # 1-index
    depot = data._depots[0]
    capacity = data._vehicle_types[0].capacity[0]
    counter = [0 for _ in range(n + 5)] # 1-index
    sum_route = [0 for _ in range(n + 5)]
    routes = [list(route) for route in sol.routes()]
    subproblems_set = set()
    for i in range(iters):
        for j in range(len(routes)):
            sum_route[j] = sum([counter[v] for v in routes[j]])
        X = np.array([(np.mean([clients[v].x for v in route]), np.mean([clients[v].y for v in route])) for route in routes]) # Depot?
        k_nearest = get_k_nearest(X, k)
        idx = None
        mn_count = 9999999
        for j in range(len(routes)):
            s = sum([sum_route[r] for r in k_nearest[j]])
            if s < mn_count:
                mn_count = s
                idx = j
        points = []
        for r in k_nearest[idx]:
            for v in routes[r]:
                points.append(v)
                counter[v] += 1
        hash_val = tuple(sorted(points))
        if hash_val in subproblems_set:
            continue
        subproblems_set.add(hash_val)
        x = [depot.x] + [clients[v].x for v in points]
        y = [depot.y] + [clients[v].y for v in points]
        demand = [0] + [clients[v].delivery[0] for v in points]
        early = [0] + [clients[v].tw_early for v in points]
        late = [46000] + [clients[v].tw_late for v in points] # TODO
        service = [0] + [clients[v].service_duration for v in points]
        m = create_problem(x, y, demand, capacity, early, late, service)
        print(f'Iterations: {i}', end=' ')
        print(f'Subproblem size: {len(x)}', end=' ')
        cost = 0
        for route in routes:
            cost += dimacs_round(np.sqrt((depot.x - clients[route[0]].x) ** 2 + (depot.y - clients[route[0]].y) ** 2))
            cost += dimacs_round(np.sqrt((depot.x - clients[route[-1]].x) ** 2 + (depot.y - clients[route[-1]].y) ** 2))
            for j in range(len(route) - 1):
                cost += dimacs_round(np.sqrt((clients[route[j]].x - clients[route[j + 1]].x) ** 2 + (clients[route[j]].y - clients[route[j + 1]].y) ** 2))
        print(f'Before: {cost}', end=' ')
        before_cost = cost
        org_routes = routes.copy()
        for r in sorted(k_nearest[idx], reverse=True):
            del routes[r]
        if True:
            new_routes, _, _ = solve_by_hgs(m, None, 1000)
            for route in new_routes:
                routes.append([points[v - 1] for v in route])
        else:
            improved_sol = m.solve(stop=MaxIterations(1000), display=False).best
            for route in improved_sol.routes():
                routes.append([points[v - 1] for v in route])
        # pdb.set_trace()
        cost = 0
        for route in routes:
            cost += dimacs_round(np.sqrt((depot.x - clients[route[0]].x) ** 2 + (depot.y - clients[route[0]].y) ** 2))
            cost += dimacs_round(np.sqrt((depot.x - clients[route[-1]].x) ** 2 + (depot.y - clients[route[-1]].y) ** 2))
            for j in range(len(route) - 1):
                cost += dimacs_round(np.sqrt((clients[route[j]].x - clients[route[j + 1]].x) ** 2 + (clients[route[j]].y - clients[route[j + 1]].y) ** 2))
        if cost > before_cost:
            cost = before_cost
            routes = org_routes
        print(f'After: {cost}')
        if start_time:
            cur_time = time.time()
            print(f'Total time cost: {cur_time - start_time}')

def write_problems(data: Model, path: str):
    clients = data._clients
    depot = data._depots[0]
    capacity = data._vehicle_types[0].capacity[0]
    n = len(clients) + 1
    with open(path, "w") as f:
        f.write("NAME : blank\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRPTW\n")
        f.write("VEHICLES : " + str(n) + "\n")
        f.write("CAPACITY : " + str(capacity) + "\n")
        f.write("DIMENSION : " + str(n) + "\nEDGE_WEIGHT_TYPE : EUC_2D\n")

        f.write("NODE_COORD_SECTION\n")
        f.write(f"1 {depot.x} {depot.y}\n")
        for l in range(n-1):
            f.write(" "+str(l+2)+" "+str(clients[l].x)+" "+str(clients[l].y)+"\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for l in range(n-1):
            f.write(str(l + 2) + " " + str(clients[l].delivery[0])+"\n")
        f.write("TIME_WINDOW_SECTION\n")
        f.write("1 0 46000\n")
        for l in range(n-1):
            f.write(str(l + 2) + " " + str(clients[l].tw_early) + " " + str(clients[l].tw_late) + "\n")
        f.write("SERVICE_TIME_SECTION\n")
        f.write("1 0\n")
        for l in range(n-1):
            f.write(str(l + 2) + " " + str(clients[l].service_duration) + '\n')
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")

def routes2str(routes):
    s = ''
    for route in routes:
        s += '0 '
        for v in route:
            s += str(v) + ' '
    s = s[:-1]
    return s

def solve_by_ls(data: Model):
    tmp_problem_path = 'data/tmp/tmp.cvrptw'
    tmp_solution_path = 'data/tmp/tmp.sol'
    write_problems(data, tmp_problem_path)
    n = len(data._clients) + 1
    cmd = f'./my_hgs_vrptw/genvrp {tmp_problem_path} {tmp_solution_path} -intensificationProbabilityLS 100 -initialSolution "'
    for j in range(1, n):
        cmd += f'0 {j}'
        if j != n - 1:
            cmd += ' '
    cmd += '"'
    os.system(cmd)
    return read_sol(tmp_solution_path)

def solve_by_hgs(data: Model, max_t=None, max_it=None):
    tmp_problem_path = 'data/tmp/tmp.cvrptw'
    tmp_solution_path = 'data/tmp/tmp.sol'
    write_problems(data, tmp_problem_path)
    cmd = f'./hgs_vrptw/genvrp {tmp_problem_path} {tmp_solution_path}'
    if max_t:
        cmd += f' -t {max_t}'
    if max_it:
        cmd += f' -it {max_it}'
    cmd += ' > out.log'
    os.system(cmd)
    return read_sol(tmp_solution_path)

def improve_by_ls(data: Model, routes):
    tmp_problem_path = 'data/tmp/tmp.cvrptw'
    tmp_solution_path = 'data/tmp/tmp.sol'
    write_problems(data, tmp_problem_path)
    cmd = f'./my_hgs_vrptw/genvrp {tmp_problem_path} {tmp_solution_path} -intensificationProbabilityLS 100 -initialSolution "'
    cmd += routes2str(routes)
    cmd += '"'
    cmd += ' > out.log'
    os.system(cmd)
    return read_sol(tmp_solution_path)