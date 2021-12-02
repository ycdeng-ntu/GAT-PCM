from core.agent_manager import AgentManager
from baselines.roda import RODAAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run RODA')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-t', '--t', type=int, required=True, help='t distance')
    parser.add_argument('-k', '--k', type=int, required=True, help='k opt')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    RODAAgent.cycle_cnt = args.cycle_cnt
    RODAAgent.t = args.t
    RODAAgent.k = args.k - 1  # minus 1 to exclude mediator itself

    am = AgentManager(args.path, RODAAgent, print_mailer_log=args.verbose)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')