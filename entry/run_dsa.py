from core.agent_manager import AgentManager
from baselines.dsa import DSAAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run DSA')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-p', '--prob', type=float, required=True, help='probability')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    DSAAgent.cycle_cnt = args.cycle_cnt
    DSAAgent.p = args.prob

    am = AgentManager(args.path, DSAAgent, print_mailer_log=args.verbose)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')