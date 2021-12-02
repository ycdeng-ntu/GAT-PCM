from core.agent_manager import AgentManager
from heuristics.full_ptisbb import PTISBBAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run PT-ISBB')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-k', '--k', type=int, required=True, help='memory budget')
    parser.add_argument('-lb', '--lb_ordering', action='store_true', help='enable LB domain ordering')
    args = parser.parse_args()

    PTISBBAgent.k = args.k
    PTISBBAgent.upper_bound_level = PTISBBAgent.ordering_level = -1
    PTISBBAgent.model = None
    PTISBBAgent.lb_ordering = args.lb_ordering

    am = AgentManager(args.path, PTISBBAgent)
    am.run()
    ncccs, elapse, _ = am.result_obj
    print(f'Simulated runtime: {elapse}; NCCCs: {ncccs}')