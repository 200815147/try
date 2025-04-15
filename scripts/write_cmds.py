import os

# for i in range(100, 150):
#     os.system(f'nohup ./my_hgs_vrptw/genvrp data/cvrptw/1000/problem/{i}.cvrptw data/cvrptw/1000/my_hgs/solution/{i}.sol -seed 0 > data/cvrptw/1000/my_hgs/log/{i}.log &')

# print('')

for i in range(900, 1000):
    os.system(f'nohup ./hgs_vrptw/genvrp data/cvrptw/1000/problem/{i}.cvrptw data/cvrptw/1000/hgs/solution/{i}.sol -seed 0 > data/cvrptw/1000/hgs/log/{i}.log &')
