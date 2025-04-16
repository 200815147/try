import os

# for i in range(100, 150):
#     os.system(f'nohup ./my_hgs_vrptw/genvrp data/cvrptw/1000/problem/{i}.cvrptw data/cvrptw/1000/my_hgs/solution/{i}.sol -seed 0 > data/cvrptw/1000/my_hgs/log/{i}.log &')

# print('')
os.makedirs('data/cvrptw/1000/hgs/solution', exist_ok=True)
os.makedirs('data/cvrptw/1000/hgs/log', exist_ok=True)
for i in range(20):
    os.system(f'nohup ./hgs_vrptw/genvrp data/cvrptw/1000/problem/{i}.cvrptw data/cvrptw/1000/hgs/solution/{i}.sol -seed 0 > data/cvrptw/1000/hgs/log/{i}.log &')
