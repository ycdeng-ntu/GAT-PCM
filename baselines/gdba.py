import random

from core.agent import Agent
import core.constants as cons

class MsgType:
    MSG_VALUE = 0
    MSG_GAIN = 1


class GDBAAgent(Agent):
    cycle_cnt = 1000

    def __init__(self, id, domain, neighbors, constraint_functions):
        super(GDBAAgent, self).__init__(id, domain, neighbors, constraint_functions)
        self.cycle_count_end = GDBAAgent.cycle_cnt
        self.pending_val = 0
        self.local_view = dict()
        self.gain_view = dict()
        self.minimum_costs = dict()
        self.eval_local_cost = True
        self.modifier = dict()
        self.gain = 0
        self.value_rcv_cnt = self.gain_rcv_cnt = 0
        self.neighbors_changed = False
        self.traffic = 0
        self.traffic_in_cycle = []
        self.ncccs = 0
        self.ncccs_in_cycle = []

    def start(self):
        self.val = random.randrange(self.domain)
        self.val_to_assign = self.val
        self.eff_init()
        self.minimum_init()
        self.send_value()

    def eff_init(self):
        for n in self.neighbors:
            m = []
            for val in range(self.domain):
                a = [0] * self.domain
                m.append(a)
            self.modifier[n] = m

    def minimum_init(self):
        for n in self.neighbors:
            mc = cons.MAX_VALUE
            for a in self.constraint_functions[n]:
                for v in a:
                    mc = min(mc, v)
            self.minimum_costs[n] = mc

    def send_value(self, go=False):
        for n in self.neighbors:
            self._send_message(n, (self.val_to_assign, go), MsgType.MSG_VALUE, (self.ncccs, self.stop_watch.elapse))

    def send_gain(self):
        for n in self.neighbors:
            self._send_message(n, self.gain, MsgType.MSG_GAIN, (self.ncccs, self.stop_watch.elapse))

    def eff_cost(self, val):
        acc = 0
        for n in self.neighbors:
            acc += self.constraint_functions[n][val][self.local_view[n]] * \
                   (self.modifier[n][val][self.local_view[n]] + 1)
            self.ncccs += 1
        return acc

    def get_local_cost(self):
        cost = super(GDBAAgent, self).get_local_cost()
        return cost

    def _dispose_message(self, msg):
        super(GDBAAgent, self)._dispose_message(msg)
        sender, content, typ = msg.sender, msg.content, msg.type
        self.stop_watch.start()
        if typ == MsgType.MSG_VALUE:
            self.traffic += 1
            nei_val, go = content
            if not self.neighbors_changed and nei_val != self.local_view.get(sender, -1):
                self.neighbors_changed = True
            if go:
                self.neighbors_changed = True
            self.local_view[sender] = nei_val
            self.value_rcv_cnt += 1
            if self.value_rcv_cnt == len(self.neighbors):
                self.val = self.val_to_assign
                self.value_rcv_cnt = 0
                costs = [self.eff_cost(val) for val in range(self.domain)]
                new_local_cost = costs[self.val]
                for val in range(self.domain):
                    if costs[val] < new_local_cost:
                        new_local_cost = costs[val]
                        self.pending_val = val

                self.gain = costs[self.val] - new_local_cost
                if self.gain <= 0 and not self.neighbors_changed:
                    for n in self.neighbors:
                        if self.constraint_functions[n][self.val][self.local_view[n]] > self.minimum_costs[n]:
                            m = self.modifier[n]
                            for a in m:
                                for i in range(len(a)):
                                    a[i] += 1
                                    self.ncccs += 1
                self.neighbors_changed = False
                self.send_gain()
        elif typ == MsgType.MSG_GAIN:
            self.round_end = True
            self.traffic += 1
            self.traffic_in_cycle.append(self.traffic)
            self.ncccs_in_cycle.append(self.ncccs)
            self.gain_rcv_cnt += 1
            self.gain_view[sender] = content
            if self.gain_rcv_cnt == len(self.neighbors):
                self.gain_rcv_cnt = 0
                self.cycle_count_end -= 1
                if self.cycle_count_end <= 0:
                    self.stop()
                    self.stop_watch.stop()
                    return
                go = True
                for n in self.neighbors:
                    if self.gain_view[n] > self.gain:
                        go = False
                        break
                if go:
                    self.val_to_assign = self.pending_val
                self.send_value(go)
        self.stop_watch.stop()

    def on_receive(self, msg, statistics):
        super(GDBAAgent, self).on_receive(msg, statistics)
        if statistics is not None:
            ncccs, elapse = statistics
            self.ncccs = max(self.ncccs, ncccs)
            self.stop_watch.update(elapse)

    def stop(self):
        super(GDBAAgent, self).stop()
        if self.mailer.result_obj is None:
            self.mailer.result_obj = (self.ncccs_in_cycle, self.traffic_in_cycle, self.stop_watch.elapse)
        else:
            nic, tic, elapse = self.mailer.result_obj

            self.mailer.result_obj = (
            [max(x, y) for x, y in zip(nic, self.ncccs_in_cycle)], [x + y for x, y in zip(tic, self.traffic_in_cycle)], max(self.stop_watch.elapse, elapse))