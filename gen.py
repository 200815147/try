import torch
from torch import Tensor
import numpy as np
import pdb
import argparse
import random
import os
from tqdm import tqdm
from utils import set_random_seed

def get_distance(x: Tensor, y: Tensor):
    """Euclidean distance between two tensors of shape `[..., n, dim]`"""
    return (x - y).norm(p=2, dim=-1)

def generate_time_windows(
        locs: torch.Tensor,
    ) -> torch.Tensor:
        """Generate time windows (TW) and service times for each location including depot.
        We refer to the generation process in "Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization"
        (Liu et al., 2024). Note that another way to generate is from "Learning to Delegate for Large-scale Vehicle Routing" (Li et al, 2021) which
        is used in "MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts" (Zhou et al, 2024). Note that however, in that case
        the distance limit would have no influence when time windows are present, since the tw for depot is the same as distance with speed=1.
        This function can be overridden for that implementation.
        See also https://github.com/RoyalSkye/Routing-MVMoE

        Args:
            locs: [B, N+1, 2] (depot, locs)

        Returns:
            time_windows: [B, N+1, 2]
            service_time: [B, N+1]
        """
        max_time = 4.6
        batch_size, n_loc = locs.shape[0], locs.shape[1] - 1  # no depot

        a, b, c = 0.15, 0.18, 0.2
        service_time = a + (b - a) * torch.rand(batch_size, n_loc)
        tw_length = b + (c - b) * torch.rand(batch_size, n_loc)
        d_0i = get_distance(locs[:, 0:1], locs[:, 1:])
        h_max = (max_time - service_time - tw_length) / d_0i - 1
        tw_start = (1 + (h_max - 1) * torch.rand(batch_size, n_loc)) * d_0i
        tw_end = tw_start + tw_length

        # Depot tw is 0, max_time
        time_windows = torch.stack(
            (
                torch.cat((torch.zeros(batch_size, 1), tw_start), -1),  # start
                torch.cat((torch.full((batch_size, 1), max_time), tw_end), -1),
            ),  # en
            dim=-1,
        )
        # depot service time is 0
        service_time = torch.cat((torch.zeros(batch_size, 1), service_time), dim=-1)
        return time_windows, service_time  # [B, N+1, 2], [B, N+1]

precision = 10000

def write_instance(locs, time_windows, service_time, instance_name, instance_filename, problem='CVRPTW'):
    with open(instance_filename, "w") as f:

        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : " + problem + "\n")
        if problem == 'CVRPTW':
            f.write("VEHICLES : " + str(len(locs)) + "\n")
        f.write("CAPACITY : " + str(int(230 + (len(locs) - 1000) / 33.3)) + "\n")
        # f.write("SERVICE_TIME : " + str(service_time * precision) + "\n" )
        f.write("DIMENSION : " + str(len(locs)) + "\nEDGE_WEIGHT_TYPE : EUC_2D\n")

        f.write("NODE_COORD_SECTION\n")
        for l in range(len(locs)):
            f.write(" "+str(l+1)+" "+str(int(locs[l][0]*precision))[:15]+" "+str(int(locs[l][1]*precision))[:15]+"\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for l in range(1, len(locs)):
            f.write(str(l + 1) + " " + str(np.random.randint(1, 11))+"\n")
        if problem == 'CVRPTW':
            f.write("TIME_WINDOW_SECTION\n")
            f.write("1 0 " + str(int(precision * time_windows[0][1])) + "\n")
            for l in range(1, len(locs)):
                f.write(str(l + 1) + " " + str(int(time_windows[l][0] * precision)) + " " + str(int(time_windows[l][1] * precision)) + "\n")
            f.write("SERVICE_TIME_SECTION\n")
            for l in range(len(service_time)):
                f.write(str(l + 1) + " " + str(int(service_time[l] * precision)) + '\n')
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")


def write_matrix_instance(n, instance_name, instance_filename):
    with open(instance_filename, "w") as f:

        f.write("NAME : " + instance_name + "\n")
        f.write("COMMENT : blank\n")
        f.write("TYPE : CVRP\n")
        f.write("DIMENSION : " + str(n) + "\n") # include depot
        f.write("CAPACITY : " + str(50) + "\n")
        f.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")
        f.write("EDGE_WEIGHT_FORMAT: FULL_MATRIX\n")
        f.write("EDGE_WEIGHT_SECTION\n")

        # 生成一个 (size, size) 的随机整数矩阵，范围是 1 到 100
        matrix = np.random.randint(1, 101, size=(n, n))
        # 使矩阵对称
        symmetric_matrix = (matrix + matrix.T) // 2
        # 将对角线元素设为 0
        np.fill_diagonal(symmetric_matrix, 0)

        for i in range(n):
            for j in range(n):
                f.write(str(symmetric_matrix[i][j]) + " ")
            f.write("\n")
        f.write("DEMAND_SECTION\n")
        f.write("1 0\n")
        for l in range(1, n):
            f.write(str(l + 1) + " " + str(np.random.randint(1, 11))+"\n")
        # if problem == 'CVRPTW':
        #     f.write("TIME_WINDOW_SECTION\n")
        #     f.write("1 0 " + str(int(precision * time_windows[0][1])) + "\n")
        #     for l in range(1, len(locs)):
        #         f.write(str(l + 1) + " " + str(int(time_windows[l][0] * precision)) + " " + str(int(time_windows[l][1] * precision)) + "\n")
        #     f.write("SERVICE_TIME_SECTION\n")
        #     for l in range(len(service_time)):
        #         f.write(str(l + 1) + " " + str(int(service_time[l] * precision)) + '\n')
        f.write("DEPOT_SECTION\n 1\n -1\n")
        f.write("EOF\n")

if __name__ == '__main__':
    set_random_seed(0)
    # write_matrix_instance(100, 'test', 'test_data/acvrp.cvrp')
    # exit(0)
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--num_loc', type=int)
    parser.add_argument('--num_instance', type=int, default=1)
    parser.add_argument('--problem', type=str, default='CVRPTW')
    args = parser.parse_args()
    bs = args.num_instance
    n = args.num_loc - 1
    locs = torch.rand(bs, n + 1, 2)
    # print(locs)
    time_windows, service_time = generate_time_windows(locs)
    # print(time_windows, service_time)
    # pdb.set_trace()
    os.makedirs(f'data/{args.problem.lower()}/{n + 1}/problem', exist_ok=True)
    for i in tqdm(range(bs)):
         write_instance(locs[i], time_windows[i], service_time[i], f'{args.problem}_{i}', f'data/{args.problem.lower()}/{n + 1}/problem/{i}.{args.problem.lower()}', args.problem)