import random

from core.agent import DFSAgent
from core.parse import parse
from queue import Queue


class AgentManager:
    def __init__(self, path, agent_type, root=None, rnd_root=False, print_mailer_log=False, print_agent_log=False, require_root=False, terminate_patience=-1):
        self.agents = parse(path, agent_type)
        self.message_queue = Queue()
        rnd_agent = list(self.agents.values())[0]
        self.cost_in_cycle = None if not rnd_agent.eval_local_cost else list()
        self.simulated_time_in_cycle = None if not rnd_agent.eval_local_cost else list()
        self.root = None
        self.terminate_patience = terminate_patience
        if isinstance(rnd_agent, DFSAgent) or require_root:
            if root is not None:
                self.root = root
            else:
                if rnd_root:
                    self.root = random.choice(list(self.agents.keys()))
                else:
                    # use the maximum degree heuristic
                    max_degree = 0
                    for ag in self.agents.values():
                        if len(ag.neighbors) > max_degree:
                            self.root = ag.id
                            max_degree = len(ag.neighbors)
        self.result_obj = None
        self.print_mailer_log = print_mailer_log
        self.print_agent_log = print_agent_log
        self.force_terminate = False
        self.lexco_ordering = list(self.agents.keys())
        self._register_mailer()

    def next(self, ag_id):
        idx = self.lexco_ordering.index(ag_id)
        if idx < len(self.lexco_ordering) - 1:
            return self.lexco_ordering[idx + 1]
        return None

    def prev(self, ag_id):
        idx = self.lexco_ordering.index(ag_id)
        if idx > 0:
            return self.lexco_ordering[idx - 1]
        return None

    def ordered_before(self, ag_id1, ag_id2):
        return self.lexco_ordering.index(ag_id1) < self.lexco_ordering.index(ag_id2)

    def _register_mailer(self):
        for ag in self.agents.values():
            ag.mailer = self
            ag.print_log = self.print_agent_log

    def post_message(self, dest, message, statistics=None):
        self.message_queue.put((dest, message, statistics))

    @property
    def agent_cnt(self):
        return len(self.agents)

    def run(self):
        terminated_agents = set()
        for ag in self.agents.values():
            ag.start()
        convergence_cycle = 0
        while len(terminated_agents) < len(self.agents) and not self.force_terminate:
            while not self.message_queue.empty():
                dest, message, statistics = self.message_queue.get()
                dest = self.agents[dest]
                dest.on_receive(message, statistics)
            cost_sum = 0
            round_ends = True
            sim_time = 0
            for ag in self.agents.values():
                cost, re = ag.timestep_advance()
                sim_time = max(sim_time, ag.stop_watch.elapse)
                if not re:
                    round_ends = False
                else:
                    assert round_ends
                    cost_sum += cost
                if ag.terminate:
                    terminated_agents.add(ag.id)
                    if self.print_mailer_log:
                        print(f'Agent {ag.id} terminates')
            if self.cost_in_cycle is not None and round_ends:
                if len(self.cost_in_cycle) > 0 and cost_sum == self.cost_in_cycle[-1]:
                    convergence_cycle += 1
                else:
                    convergence_cycle = 0
                self.cost_in_cycle.append(cost_sum)
                self.simulated_time_in_cycle.append(sim_time)
                if self.print_mailer_log:
                    print(f'Cycle {len(self.cost_in_cycle)}\t Simulated time {self.simulated_time_in_cycle[-1]:.2f}: {self.cost_in_cycle[-1]}')
                if self.terminate_patience == convergence_cycle:
                    for ag in self.agents.values():
                        ag.stop()
                    print('Early stop')
                    break
        if self.print_mailer_log:
            print('Terminated')