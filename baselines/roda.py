import random

import torch

from core.agent import Agent
from random import randrange, sample
from core.bucket import Bucket


class RODAAgent(Agent):
    t = 2
    k = 2
    cycle_cnt = 1000

    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.cycle = 0
        self.t_hop_neighbors = set()
        self.msg_rcv_cnt = dict()
        self.region = set()
        self.region_info = dict()
        self.pending_val = -1
        self.pending_med = None
        self.gain = 0
        self.all_assign_info = dict()
        self.confirm_region = set()
        self.confirm_med = set()
        self.eval_local_cost = True
        self.traffic = 0
        self.traffic_in_cycle = []
        self.stalled = False
        self.region_constraints = dict()

    def start(self):
        self.val = randrange(self.domain)
        self.t_hop_neighbors.update(self.neighbors)
        self.msg_rcv_cnt['HOP'] = 0
        self.aug_neighbors = set(self.neighbors)
        self.aug_neighbors.add(self.id)
        self.region_constraints[self.id] = self.constraint_functions
        for n in self.neighbors:
            self._send_message(n, self.val, 'VAL')
            self._send_message(n, ({self.id}, RODAAgent.t, self.region_constraints), 'HOP')

    def _dispose_message(self, msg):
        typ, sender, content = msg.type, msg.sender, msg.content
        self.stop_watch.start()
        if typ == 'VAL':
            self.local_view[sender] = content
            self.traffic += 1
        elif typ == 'HOP':
            neighbors, ttl, region_constraints = content
            for cf in region_constraints.values():
                self.traffic += len(cf) * int(self.domain ** 2)
            self.traffic += len(neighbors) + 1
            self.region_constraints.update(region_constraints)
            self.t_hop_neighbors.update(neighbors)
            ttl -= 1
            self.msg_rcv_cnt['HOP'] += 1
            if self.msg_rcv_cnt['HOP'] == len(self.neighbors):
                self.msg_rcv_cnt['HOP'] = 0
                if ttl > 0:
                    for n in self.neighbors:
                        self._send_message(n, (set(self.t_hop_neighbors), ttl, dict(self.region_constraints)), 'HOP')
                else:
                    self.t_hop_neighbors.discard(self.id)
                    self.stop_watch.stop()
                    self.start_roda_cycle()
        elif typ == 'AGG':
            region_info, ttl = content
            for lv, _ in region_info.values():
                self.traffic += len(lv) + 1
            self.region_info.update(region_info)
            self.msg_rcv_cnt['AGG'] += 1
            if self.msg_rcv_cnt['AGG'] == len(self.neighbors):
                self.msg_rcv_cnt['AGG'] = 0
                ttl -= 1
                if ttl > 0:
                    for n in self.neighbors:
                        self._send_message(n, (dict(self.region_info), ttl), 'AGG', self.stop_watch.elapse)
                else:
                    if not self.stalled:
                        dom, functions, old_assign, cost = self.compile()
                        assign, new_cost = self.solve(dom, functions)
                        self.pending_val = assign[self.id]
                        old_cost = self.get_cost(functions, old_assign)
                        gain = old_cost - new_cost
                    else:
                        gain = random.randint(0, 100)
                        old_assign = dict()
                        for n in self.region:
                            local_view, n_val = self.region_info[n]
                            old_assign[n] = n_val
                        assign = dict(old_assign)
                    self.pending_med = self.id
                    self.all_assign_info.clear()
                    self.all_assign_info[self.id] = (assign, gain, self.region)
                    self.msg_rcv_cnt['CONTEST'] = 0
                    for n in self.neighbors:
                        self._send_message(n, (dict(self.all_assign_info), RODAAgent.t), 'CONTEST', self.stop_watch.elapse)
        elif typ == 'CONTEST':
            assign_info, ttl = content
            for a, _, r in assign_info.values():
                self.traffic += len(a) + 1 + len(r)

            self.all_assign_info.update(assign_info)
            self.msg_rcv_cnt['CONTEST'] += 1
            if self.msg_rcv_cnt['CONTEST'] == len(self.neighbors):
                self.msg_rcv_cnt['CONTEST'] = 0
                ttl -= 1
                if ttl > 0:
                    for n in self.neighbors:
                        self._send_message(n, (dict(self.all_assign_info), ttl), 'CONTEST', self.stop_watch.elapse)
                else:
                    max_gain = -1
                    max_gain_med = None
                    for med in self.all_assign_info.keys():
                        _, gain, region = self.all_assign_info[med]
                        if self.id in region:
                            if gain > max_gain:
                                max_gain = gain
                                max_gain_med = med
                            elif gain == max_gain and max_gain_med > med:
                                max_gain_med = med
                    self.confirm_region.clear()
                    self.confirm_region.add((max_gain_med, self.id))
                    self.msg_rcv_cnt['CONFIRM'] = 0
                    for n in self.neighbors:
                        self._send_message(n, (set(self.confirm_region), RODAAgent.t), 'CONFIRM', self.stop_watch.elapse)
        elif typ == 'CONFIRM':
            confirm_region, ttl = content
            self.traffic += len(confirm_region)
            self.confirm_region.update(confirm_region)
            self.msg_rcv_cnt['CONFIRM'] += 1
            if self.msg_rcv_cnt['CONFIRM'] == len(self.neighbors):
                self.msg_rcv_cnt['CONFIRM'] = 0
                ttl -= 1
                if ttl > 0:
                    for n in self.neighbors:
                        self._send_message(n, (set(self.confirm_region), ttl), 'CONFIRM', self.stop_watch.elapse)
                else:
                    confirm_cnt = 0
                    for med, ag in self.confirm_region:
                        if med == self.id:
                            confirm_cnt += 1
                    self.confirm_med.clear()
                    if confirm_cnt == len(self.region):
                        self.confirm_med.add(self.id)
                    for n in self.aug_neighbors:
                        self._send_message(n, (set(self.confirm_med), RODAAgent.t), 'NOTIFY', self.stop_watch.elapse)
                    self.msg_rcv_cnt['NOTIFY'] = 0
        elif typ == 'NOTIFY':
            confirmed_med, ttl = content
            self.traffic += len(confirmed_med)
            self.confirm_med.update(confirmed_med)
            self.msg_rcv_cnt['NOTIFY'] += 1
            if self.msg_rcv_cnt['NOTIFY'] == len(self.aug_neighbors):
                self.msg_rcv_cnt['NOTIFY'] = 0
                ttl -= 1
                if ttl > 0:
                    for n in self.aug_neighbors:
                        self._send_message(n, (set(self.confirm_med), ttl), 'NOTIFY', self.stop_watch.elapse)
                else:
                    assigned = False
                    for med in self.confirm_med:
                        assign, _, region = self.all_assign_info[med]
                        if self.id in region:
                            assert not assigned
                            self.val = assign[self.id]
                            assigned = True
                    for n in self.neighbors:
                        self._send_message(n, self.val, 'UPDATE', self.stop_watch.elapse)
                    self.msg_rcv_cnt['UPDATE'] = 0
                    # assert assigned

        elif typ == 'UPDATE':
            self.traffic += 1
            self.local_view[sender] = content
            self.msg_rcv_cnt['UPDATE'] += 1
            if self.msg_rcv_cnt['UPDATE'] == len(self.neighbors):
                self.cycle += 1
                if self.cycle <= RODAAgent.cycle_cnt:
                    self.stop_watch.stop()
                    self.start_roda_cycle()
                else:
                    self.stop()
        self.stop_watch.stop()

    def compile(self, debug=False):
        dom = dict()
        functions = dict()
        cost = 0
        processed = set()
        old_assign = dict()
        for n in self.region:
            local_view, n_val = self.region_info[n]
            constraints = self.region_constraints[n]
            functions[n] = dict()
            old_assign[n] = n_val
            for n_prime, matrix_prime in constraints.items():
                tag = tuple(sorted([n, n_prime]))
                if tag not in processed:
                    processed.add(tag)
                    if debug:
                        if n_prime not in self.region:
                            print(n, matrix_prime[n_val][local_view[n_prime]], sep='\t')
                        else:
                            print(tag, matrix_prime[n_val][local_view[n_prime]], sep='\t')
                    cost += matrix_prime[n_val][local_view[n_prime]]
                if n not in dom:
                    dom[n] = len(matrix_prime)
                if n_prime not in self.region:
                    matrix = [matrix_prime[val][local_view[n_prime]] for val in range(dom[n])]
                    if n not in functions[n]:
                        functions[n][n] = matrix
                    else:
                        old_vec = functions[n][n]
                        functions[n][n] = [x + y for x, y in zip(old_vec, matrix)]
                else:
                    functions[n][n_prime] = matrix_prime
        return dom, functions, old_assign, cost

    @staticmethod
    def get_cost(functions, assigns, debug=False):
        cost = 0
        processed = set()
        for var in functions.keys():
            for n in functions[var].keys():
                tag = tuple(sorted([var, n]))
                if tag in processed:
                    continue
                processed.add(tag)
                if n == var:
                    cost += functions[var][n][assigns[var]]
                    if debug:
                        print(n, functions[var][n][assigns[var]], sep='\t')
                else:
                    if debug:
                        print(tag, functions[var][n][assigns[var]][assigns[n]], sep='\t')
                    cost += functions[var][n][assigns[var]][assigns[n]]
        return cost

    @staticmethod
    def solve(dom, functions):
        var_ordering = list(dom.keys())
        bucket = None
        buckets = []
        for var in var_ordering:
            local_buckets = [Bucket.from_matrix(functions[var][n], var, n) for n in functions[var].keys()
                             if var_ordering.index(n) > var_ordering.index(var)]
            if var in functions[var]:
                local_buckets.append(Bucket(torch.tensor(functions[var][var]), [var]))
            if bucket is not None:
                local_buckets.append(bucket)
            bucket = Bucket.join(local_buckets)
            buckets.append(bucket)
            bucket = bucket.proj(var)
        assign = dict()
        for var, bucket in zip(reversed(var_ordering), reversed(buckets)):
            bucket = bucket.reduce(assign)
            assert len(bucket.dims) == 1
            val = bucket.data.argmin().item()
            assign[var] = val
        return assign, buckets[-1].data.min().item()

    def start_roda_cycle(self):
        self.round_end = True
        self.traffic_in_cycle.append(self.traffic)
        k = min(len(self.t_hop_neighbors), RODAAgent.k)
        self.region = sample(self.t_hop_neighbors, k)
        self.region.append(self.id)
        self.region_info.clear()
        self.region_info[self.id] = (self.local_view, self.val)
        self.msg_rcv_cnt['AGG'] = 0
        for n in self.neighbors:
            self._send_message(n, (dict(self.region_info), RODAAgent.t), 'AGG', self.stop_watch.elapse)

    def on_receive(self, msg, statistics):
        super(RODAAgent, self).on_receive(msg, statistics)
        if statistics is not None:
            self.stop_watch.update(statistics)

    def stop(self):
        super(RODAAgent, self).stop()
        # print(self.traffic_in_cycle)
        if self.mailer.result_obj is None:
            self.mailer.result_obj = self.traffic_in_cycle
        else:
            tic = self.mailer.result_obj
            self.mailer.result_obj = [x + y for x, y in zip(tic, self.traffic_in_cycle)]
