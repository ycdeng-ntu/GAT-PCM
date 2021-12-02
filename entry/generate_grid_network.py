import os

from core.problem import Problem
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate grid problems')
    parser.add_argument('-g', '--grid_size', type=int, required=True, help='grid size')
    parser.add_argument('-d', '--domain_size', type=int, required=True, help='domain size')
    parser.add_argument('-o', '--output', type=str, required=True, help='output dir')
    parser.add_argument('-n', '--num_instances', type=int, required=False, default=1, help='number of instance to generate')
    args = parser.parse_args()
    for i in range(args.num_instances):
        p = Problem()
        p.random_sensor_net(args.grid_size, args.domain_size)
        p.save(os.path.join(args.output, f'{i}.xml'))