from core.agent_manager import AgentManager
from baselines.max_sum_advp import MaxSumADVPAgent
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run Max-sum_ADVP')
    parser.add_argument('-pth', '--path', type=str, required=True, help='problem path')
    parser.add_argument('-c', '--cycle_cnt', type=int, required=True, help='cycle count')
    parser.add_argument('-pl', '--pl', type=int, required=True, help='phase length')
    parser.add_argument('-v', '--verbose', action='store_true', help='print detailed log')
    args = parser.parse_args()

    MaxSumADVPAgent.cycle_cnt = args.cycle_cnt
    MaxSumADVPAgent.phase_length = args.pl

    am = AgentManager(args.path, MaxSumADVPAgent, print_mailer_log=args.verbose)
    am.run()
    print(f'Simulated runtime: {am.simulated_time_in_cycle[-1]}; Best cost: {min(am.cost_in_cycle)}')