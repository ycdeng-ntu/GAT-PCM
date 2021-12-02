import torch

from core.agent_manager import AgentManager
from heuristics.dlns_agent import DLNSAgent
import heuristics.gnn_embedding_utils as geu
import argparse

from model import GATNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GAT-PCM-DLNS')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-p', '--prob', type=float, required=True, help='destroy probability')
    parser.add_argument('-mp', '--model_path', type=str, required=False, help='model path', default='pretrained_model/300_54_4100.pth')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cpu')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    geu.device = torch.device(args.device)

    model = GATNet(4, 16)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.double()
    model.to(geu.device)
    DLNSAgent.model = model

    DLNSAgent.cycle_cnt = args.cycle_cnt
    DLNSAgent.p = args.prob

    am = AgentManager(args.path, DLNSAgent, print_mailer_log=args.verbose, require_root=True)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')