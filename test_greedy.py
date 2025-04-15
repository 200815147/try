from utils import *

def read_tsp_file(file_path):
    capacity = None
    distance_matrix = None
    demand = None
    dimension = None
    in_edge_weight_section = False
    in_demand_section = False
    weight_data = []
    demand_data = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('DIMENSION'):
                dimension = int(line.split(':')[1].strip())
            elif line.startswith('EDGE_WEIGHT_SECTION'):
                in_edge_weight_section = True
            elif line.startswith('DEMAND_SECTION'):
                in_demand_section = True
                in_edge_weight_section = False
            elif line.startswith('EOF'):
                break
            elif in_edge_weight_section:
                weights = [int(x) for x in line.split()]
                weight_data.extend(weights)
            elif in_demand_section:
                node, dem = map(int, line.split())
                demand_data.append(dem)
            elif line.startswith('CAPACITY'):
                capacity = int(line.split()[-1])
            if len(demand_data) == dimension:
                break

    if dimension is not None:
        distance_matrix = np.array(weight_data).reshape((dimension, dimension))
    if demand_data:
        demand = np.array(demand_data)

    return distance_matrix, demand, capacity


distance_matrix, demand, capacity = read_tsp_file('test_data/acvrp.cvrp')
tour, cost = greedy_nearest(distance_matrix, demand, capacity)
pdb.set_trace()