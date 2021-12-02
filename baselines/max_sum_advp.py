from core.agent import Agent
from core.utilities import argmin
import core.constants as cons


class MsgType:
    MSG_R = 1
    MSG_VALUE = 2


class MaxSumADVPAgent(Agent):
    phase_length = 50
    cycle_cnt = 5000

    def __init__(self, id, domain, neighbors, constraint_functions):
        super(MaxSumADVPAgent, self).__init__(id, domain, neighbors, constraint_functions)
        self.cycle = MaxSumADVPAgent.cycle_cnt
        self.eval_local_cost = True
        self.income_messages = dict()
        self.prev = set()
        self.succ = set()
        self.phase_cycle = 0
        self.phase_cnt = 0
        self.traffic = 0
        self.traffic_in_cycle = []
        self.val_rcv_cnt = 0

    def start(self):
        for neighbor in self.neighbors:
            self.income_messages[neighbor] = [0] * self.domain
            if neighbor < self.id:
                self.prev.add(neighbor)
            else:
                self.succ.add(neighbor)
        self.compute()

    def _dispose_message(self, msg):
        super(MaxSumADVPAgent, self)._dispose_message(msg)
        if msg.type == MsgType.MSG_VALUE:
            self.local_view[msg.sender] = msg.content
            self.round_end = True
            self.val_rcv_cnt += 1
            if self.val_rcv_cnt == len(self.neighbors):
                self.val_rcv_cnt = 0
                self.traffic_in_cycle.append(self.traffic)
        elif msg.type == MsgType.MSG_R:
            self.traffic += 2 * len(msg.content)
            self.income_messages[msg.sender] = msg.content

    def _on_timestep_advanced(self):
        self.cycle -= 1
        if self.cycle == 0:
            self.stop()
            return
        self.compute()

    def compute(self):
        self.stop_watch.start()
        costs = [0] * self.domain
        for msg in self.income_messages.values():
            for i in range(self.domain):
                costs[i] += msg[i]
        val = argmin(costs)
        if val != self.val:
            self.val = val
        for neighbor in self.neighbors:
            self._send_message(neighbor, val, MsgType.MSG_VALUE, statistics=self.stop_watch.elapse)
        self.phase_cycle += 1
        self.cycle -= 1
        if self.phase_cycle == MaxSumADVPAgent.phase_length:
            self.phase_cycle = 0
            tmp = self.succ
            self.succ = self.prev
            self.prev = tmp
            self.phase_cnt += 1
        if self.cycle <= 0:
            self.stop()
            self.stop_watch.stop()
            return
        for neighbor in self.succ:
            agg = [0] * self.domain
            for n in self.neighbors:
                if n != neighbor:
                    for i in range(self.domain):
                        agg[i] += self.income_messages[n][i]
            costs = []
            for val in range(self.domain):
                best_cost = cons.MAX_VALUE
                if self.phase_cnt >= 2:
                    c = self.constraint_functions[neighbor][self.val][val] + agg[self.val]
                    best_cost = c
                else:
                    for i in range(self.domain):
                        c = self.constraint_functions[neighbor][i][val] + agg[i]
                        best_cost = min(best_cost, c)
                costs.append(best_cost)
            alpha = int(sum(costs) / len(costs))
            costs = [x - alpha for x in costs]
            self._send_message(neighbor, costs, MsgType.MSG_R, self.stop_watch.elapse)
        self.stop_watch.stop()

    def on_receive(self, msg, statistics):
        super(MaxSumADVPAgent, self).on_receive(msg, statistics)
        self.stop_watch.update(statistics)

    def stop(self):
        super(MaxSumADVPAgent, self).stop()
        if self.mailer.result_obj is None:
            self.mailer.result_obj = self.traffic_in_cycle
        else:
            tic = self.mailer.result_obj
            self.mailer.result_obj = [x + y for x, y in zip(tic, self.traffic_in_cycle)]