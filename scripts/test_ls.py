import os

n = 150
m = 20
# n = 1000
# m = 1

# for i in range(m):
#     os.system(f'./hgs_vrptw/genvrp data/cvrptw/{n}/problem/{i}.cvrptw data/tmp/hgs_{i}.sol')
for i in range(m):
    cmd = f'./my_hgs_vrptw/genvrp data/cvrptw/{n}/problem/{i}.cvrptw data/tmp/ls_{i}.sol -intensificationProbabilityLS 100 -initialSolution "'
    for j in range(1, n):
        cmd += f'0 {j}'
        if j != n - 1:
            cmd += ' '
    cmd += '"'
    # print(cmd)
    # exit(0)
    os.system(cmd)