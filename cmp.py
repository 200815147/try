from utils import *

better = 0
dir1 = 'data/cvrptw/1000/my_hgs/solution'
dir2 = 'data/cvrptw/1000/hgs/solution'

for i in range(100):
    file_path1 = os.path.join(dir1, f'{i}.sol')
    file_path2 = os.path.join(dir2, f'{i}.sol')
    _, cost1, time1 = read_sol(file_path1)
    _, cost2, time2 = read_sol(file_path2)
    print(cost1, time1, " |", cost2, time2, end=" | ")
    if cost1 <= cost2:
        better += 1
        print(f'improve: {(cost2 - cost1) / cost2}')
    else:
        print(f'drop: {(cost1 - cost2) / cost1}')
print(f'Better: {better}')
