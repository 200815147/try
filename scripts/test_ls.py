import os

n = 200
m = 20
n = 1000
m = 1

# for i in range(m):
#     # pass
#     os.system(f'./hgs_vrptw/genvrp test_data/n_{n}_{i}.cvrptw hgs_{i}.sol')
for i in range(m):
    cmd = f'./my_hgs_vrptw/genvrp test_data/n_{n}_{i}.cvrptw ls_{i}.sol -initialSolution "'
    for j in range(1, n):
        cmd += f'0 {j}'
        if j != n - 1:
            cmd += ' '
    cmd += '"'
    # print(cmd)
    # exit(0)
    os.system(cmd)