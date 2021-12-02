from core.agent_manager import AgentManager
from baselines.gdba import GDBAAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run GDBA')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    GDBAAgent.cycle_cnt = args.cycle_cnt

    am = AgentManager(args.path, GDBAAgent, print_mailer_log=args.verbose)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')