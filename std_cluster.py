from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime
import argparse
import pdb
from sklearn.cluster import KMeans
import numpy as np
import time
import math
import kmedoids
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from itertools import combinations
from utils import *

MY_SEED = 200815147

# 核心计算函数
def compute_element(pair, similarity):
    i, j = pair
    val = min(similarity(i, j), similarity(j, i))
    return (i, j, val)

def parallel_matrix_build(n, similarity):
    # 初始化空矩阵（对称矩阵）
    matrix = np.zeros((n-1, n-1), dtype=np.float32)
    
    # 生成所有需要计算的(i,j)对（仅上三角部分）
    indices = list(combinations(range(1, n), 2))  # 生成C(n,2)组合
    
    # 进程池配置（根据CPU核心数调整）
    num_cores = max(1, mp.cpu_count() - 2)  # 留出2个核心给系统
    chunk_size = len(indices) // (num_cores * 4)  # 动态分块

    with mp.Pool(processes=num_cores) as pool:
        # 使用imap_unordered减少内存占用
        results = pool.imap_unordered(
            partial(compute_element, similarity=similarity),
            indices,
            chunksize=chunk_size
        )
        
        # 实时填充矩阵
        for i, j, val in results:
            matrix[i-1][j-1] = val  # 仅填充上三角
            matrix[j-1][i-1] = val   # 对称赋值下三角

    return matrix
    
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--path', type=str)
    parser.add_argument('--subproblem_size', type=int, default=100)
    args = parser.parse_args()
    _st = time.time()
    x, y, demand, capacity, early, late, service, depot_id, n = read_vrptw_problem(args.path)
    def similarity(i, j):
        distance = math.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)
        f = late[j] - (early[i] + service[i] + distance)
        h = max(early[j] - (late[i] + service[i] + distance), 0)
        return distance * (2 - (f - h) / (late[depot_id] - early[depot_id]) + (demand[i] + demand[j]) / capacity)
    set_random_seed(MY_SEED)

    matrix = np.zeros((n - 1, n - 1), dtype=float)
    if True:
        matrix = parallel_matrix_build(n, similarity)
    else:
        for i in range(1, n):
            for j in range(i + 1, n):
                matrix[i - 1][j - 1] = matrix[j - 1][i - 1] = min(similarity(i, j), similarity(j, i))
    num_subproblems = (n + args.subproblem_size - 1) // args.subproblem_size
    print(f'Num of subproblems: {num_subproblems}')
    st = time.time()
    fp = kmedoids.fasterpam(matrix, num_subproblems, random_state=MY_SEED)
    ed = time.time()
    print(f'Cluster time: {ed - st}')
    print(f'Total time: {ed - _st}')
    # n = 10000 
    # parallel
    # Cluster time: 27.523533582687378
    # Total time: 174.26852250099182
    # 
    # Cluster time: 22.244901657104492
    # Total time: 385.7727949619293
    # exit(0)
    # pdb.set_trace()
    labels = fp.labels # numpy.ndarray
    if False:
        routes, cost, _ = read_sol('data/cvrptw/1000/hgs/solution/4.sol')
        centroid = [None for _ in range(len(routes))]
        for i, route in enumerate(routes):
            xx = []
            yy = []
            for v in route:
                xx.append(x[v])
                yy.append(y[v])
            centroid[i] = (np.mean(xx), np.mean(yy))
        kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto", init="k-means++").fit(centroid)
        num_subproblems = len(routes)
        for i, route in enumerate(routes):
            for v in route:
                labels[v - 1] = kmeans.labels_[i]
        print(f'Sol cost: {cost}')
        # Cost: 1158810
        #   solver  cluster
        # 0 1168372 1167704
        # 1 1162330 1167772
        # 2 1144044 1144080
        # 3 1462459 1461928
    x_list = [[] for _ in range(num_subproblems)]
    y_list = [[] for _ in range(num_subproblems)]
    demand_list = [[] for _ in range(num_subproblems)]
    early_list = [[] for _ in range(num_subproblems)]
    late_list = [[] for _ in range(num_subproblems)]
    service_list = [[] for _ in range(num_subproblems)]
    clusters = [[] for _ in range(num_subproblems)]
    for i in range(num_subproblems):
        x_list[i].append(x[depot_id])
        y_list[i].append(y[depot_id])
        demand_list[i].append(demand[depot_id])
        early_list[i].append(early[depot_id])
        late_list[i].append(late[depot_id])
        service_list[i].append(service[depot_id])
    for i, label in enumerate(labels):
        clusters[label].append(i)
        x_list[label].append(x[i + 1])
        y_list[label].append(y[i + 1])
        demand_list[label].append(demand[i + 1])
        early_list[label].append(early[i + 1])
        late_list[label].append(late[i + 1])
        service_list[label].append(service[i + 1])
    # pdb.set_trace()
    # for i in range(num_subproblems):
    #     print(f'cluster {i} : ', end='')
    #     for j in clusters[i]:
    #         print(j + 2, end=' ')
    #     print('')
    # routes, _, _ = read_sol('hgs_1000.sol')
    # for route in routes:
    #     tmp = [int(labels[v - 1]) for v in route]
    #     # tmp = sorted(tmp)
    #     print(tmp)
    # exit(0)
    subproblems = []
    for i in range(num_subproblems):
        subproblems.append(create_problem(x_list[i], y_list[i], demand_list[i], capacity, early_list[i], late_list[i], service_list[i]))
    total_cost = 0
    sols = []
    for subproblem in subproblems:
        sol = subproblem.solve(stop=MaxIterations(1000), display=False).best
        sols.append(sol)
        cost = sol.distance()
        # pdb.set_trace()
        total_cost += cost
        # print(cost)
        # print_routes(sol)
    print(f'Total cost: {total_cost}')
    # exit(0)
    for i in range(1):
        sol, m = merge_subproblems(subproblems, sols)
        # print_routes(sol)
        print(f'Iterations: {i} Cost: {sol.distance()}')
        cur_time = time.time()
        print(f'Total time cost: {cur_time - start_time}')
        break
        # exit(0)
        subproblems = devide_subproblems(sol, args.subproblem_size * 4, m)
        sols = improve_subproblems(subproblems)
    cur_time = time.time()
    print(f'Total time cost: {cur_time - start_time}')
    improve_one_subproblem(sol, m, start_time)
# python std_cluster.py --path test_data/n_1000_0.cvrptw
# python std_cluster.py --path test_data/n_50_0.cvrptw
# python std_cluster.py --path data/cvrptw/1000/problem/0.cvrptw