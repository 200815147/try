from pyvrp import Model, read
from pyvrp.stop import MaxIterations, MaxRuntime
import argparse
import pdb


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', type=str)
args = parser.parse_args()


INSTANCE = read(args.path, round_func="dimacs")
# BKS = read_solution("data/RC208.sol")

model = Model.from_data(INSTANCE)
# result = model.solve(stop=MaxIterations(10), seed=42, display=False)
result = model.solve(stop=MaxRuntime(1200), seed=42, display=False)
print(result)
# pdb.set_trace()