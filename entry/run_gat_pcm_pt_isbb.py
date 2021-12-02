import torch

from core.agent_manager import AgentManager
from heuristics.full_ptisbb import PTISBBAgent
import heuristics.gnn_embedding_utils as geu
import argparse

from model import GATNet

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GAT-PCM-PT-ISBB')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-k', '--k', type=int, required=True, help='memory budget')
    parser.add_argument('-mp', '--model_path', type=str, required=False, help='model path',
                        default='pretrained_model/300_54_4100.pth')
    parser.add_argument('-d', '--device', type=str, required=False, help='computing device', default='cpu')
    args = parser.parse_args()

    geu.device = torch.device(args.device)

    model = GATNet(4, 16)
    model.load_state_dict(torch.load(args.model_path, map_location=args.device))
    model.double()
    model.to(geu.device)

    PTISBBAgent.k = args.k
    PTISBBAgent.upper_bound_level = -1
    PTISBBAgent.ordering_level = 2
    PTISBBAgent.model = model
    PTISBBAgent.lb_ordering = False

    am = AgentManager(args.path, PTISBBAgent)
    am.run()
    ncccs, elapse, _ = am.result_obj
    print(f'Simulated runtime: {elapse}; NCCCs: {ncccs}')