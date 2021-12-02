from core.agent_manager import AgentManager
from baselines.dlns_dpop import DLNSAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run T-DLNS')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-p', '--prob', type=float, required=True, help='destroy probability')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    DLNSAgent.cycle_cnt = args.cycle_cnt
    DLNSAgent.p = args.prob

    am = AgentManager(args.path, DLNSAgent, print_mailer_log=args.verbose, require_root=True)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')