import random

from core.agent import Agent
from core.utilities import argmin


class MsgType:
    MSG_VALUE = 0


class DSAAgent(Agent):
    p = 0.8
    cycle_cnt = 100

    def __init__(self, id, domain, neighbors, constraint_functions):
        super(DSAAgent, self).__init__(id, domain, neighbors, constraint_functions)
        self.cycle = DSAAgent.cycle_cnt
        self.eval_local_cost = True
        self.traffic = 0
        self.traffic_in_cycle = []
        self.ncccs = 0
        self.ncccs_in_cycle = []


    def start(self):
        self.val = random.randrange(0, self.domain)
        self._send_values()

    def _send_values(self):
        for neighbor in self.neighbors:
            self._send_message(neighbor, self.val, MsgType.MSG_VALUE, (self.stop_watch.elapse, self.ncccs))

    def _dispose_message(self, msg):
        super(DSAAgent, self)._dispose_message(msg)
        if msg.type == MsgType.MSG_VALUE:
            self.traffic += 1
            self.local_view[msg.sender] = msg.content

    def _on_timestep_advanced(self):
        self.stop_watch.start()
        self.cycle -= 1
        self.round_end = True
        if self.cycle <= 0:
            self.stop()
            self.stop_watch.stop()
            return
        if random.random() < DSAAgent.p:
            costs = [sum([self.constraint_functions[n][val][self.local_view[n]] for n in self.neighbors])
                     for val in range(self.domain)]
            self.ncccs += len(self.neighbors) * self.domain
            self.val = argmin(costs)
            self._send_values()
        self.stop_watch.stop()
        self.traffic_in_cycle.append(self.traffic)
        self.ncccs_in_cycle.append(self.ncccs)
        if Agent.tracer_id is not None and Agent.tracer_id == self.id and Agent.on_round_ends is not None:
            Agent.on_round_ends()

    def on_receive(self, msg, statistics):
        super(DSAAgent, self).on_receive(msg, statistics)
        elapse, ncccs = statistics
        self.stop_watch.update(elapse)
        self.ncccs = max(self.ncccs, ncccs)

    def stop(self):
        super(DSAAgent, self).stop()
        if self.mailer.result_obj is None:
            self.mailer.result_obj = (self.ncccs_in_cycle, self.traffic_in_cycle, self.stop_watch.elapse)
        else:
            nic, tic, elapse = self.mailer.result_obj

            self.mailer.result_obj = ([max(x, y) for x, y in zip(nic, self.ncccs_in_cycle)], [x + y for x, y in zip(tic, self.traffic_in_cycle)], max(self.stop_watch.elapse, elapse))