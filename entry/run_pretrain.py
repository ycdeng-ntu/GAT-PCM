import argparse
import os

from pretrain.gat_pcm_pretrain import GATPCM
from model import GATNet
from pretrain.env import Environment


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run pretrain')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem dir path')
    parser.add_argument('-c', '--cap', type=int, required=False, default=1000000, help='memory budget')
    parser.add_argument('-b', '--batch_size', type=int, required=False, default=64, help='batch size')
    parser.add_argument('-e', '--epoches', type=int, required=False, default=5000, help='number of epoches')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=10, help='number of training iterations')
    parser.add_argument('-mp', '--model_path', type=str, required=True, help='path to save models')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cuda')
    args = parser.parse_args()

    problems = []
    base = args.path
    for f in os.listdir(base):
        if f.endswith('.xml'):
            problems.append(os.path.join(base, f))
    env = Environment(problems, None)
    model = GATNet(4, 16)
    agent = GATPCM(env, model, x_init_feature=[0, 0, 1, 0], c_init_feature='[0, 1, 0, {}]',
                     f_init_feature=[1, 0, 0, 0], device=args.device, capacity=args.cap, model_path=args.model_path,
                     batch_size=args.batch_size, episode=args.epoches, iteration=args.iterations)
    agent.train()