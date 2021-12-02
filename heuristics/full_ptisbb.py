from core.agent import DFSAgent
from core.bucket import Bucket
from core.constants import MAX_VALUE
from .micro_agent import MicroAgent


class PTISBBAgent(DFSAgent):
    k = 2
    ordering_level = -1
    upper_bound_level = -1
    model = None
    lb_ordering = False

    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.ncccs = 0
        self.sep_ordering = []
        self.sep_level = dict()
        self.children_utils = dict()
        self.cpa = dict()
        self.complete_vals = set()
        self.domain_vals = []
        self.high_cost = []
        self.srch_val_idx = dict()
        self.lb_c = dict()
        self.ub = MAX_VALUE
        self.backtrack_rcv_cnt = []
        self.children_ready_cnt = 0
        self.micro_agent = None
        self.cur_var = False
        self.desc_cost = 0
        self.desc_cnt = 0
        self.assign = None
        self.bound_enforced_val = -1

    def _on_pseudo_tree_created(self):
        if self.root_agent:
            for c in self.children:
                self.send_message(c, {self.id: self.level}, 'LEVEL')
        else:
            self.send_sep_level()

    def on_micro_agent_stop(self):
        not_reset = False
        if self.micro_agent.raw_values:
            if self.micro_agent.active:
                idxes = self.micro_agent.values.argsort()
                self.domain_vals = [int(x) for x in idxes]
                if self.level <= PTISBBAgent.upper_bound_level and not self.leaf_agent:
                    self.create_micro_agent(self.cpa, False)
                    not_reset = True
                    self.micro_agent.active = True
                    self.cur_var = True
                else:
                    self.continue_assign()
        else:
            self.val = self.micro_agent.val
            self.debug(f'{self.id}: {self.val}')
            if self.cur_var:
                for c in self.children:
                    self.send_message(c, {self.id: self.val}, 'ASSIGN')
            elif self.assign is not None:
                self.dispose_assign()
        self.debug(f'{self.id} micro agent stop')
        if not not_reset:
            self.micro_agent = None

    def _dispose_message(self, msg):
        super(PTISBBAgent, self)._dispose_message(msg)
        typ, sender, content = msg.type, msg.sender, msg.content
        self.stop_watch.start()
        if typ == 'LEVEL':
            self.sep_level = {x: content[x] for x in self.sep}
            self.send_sep_level()
        elif typ == 'UTIL':
            self.children_utils[sender] = content
            if len(self.children_utils) == len(self.children):
                if not self.root_agent:
                    self.inference()
                else:
                    self.init_vars()
                    if self.level <= PTISBBAgent.ordering_level:
                        self.create_micro_agent({}, True)
                        self.micro_agent.active = True
                    elif self.level <= PTISBBAgent.upper_bound_level:
                        self.create_micro_agent({}, False)
                        self.micro_agent.active = True
                        self.cur_var = True
                    else:
                        for c in self.children:
                            self.srch_val_idx[c] = 0
                            self.send_message(c, ({self.id: self.domain_vals[0]}, MAX_VALUE), 'CPA')
        elif typ == 'BUILD_MICRO_AGENT':
            cpa = {x: content[0][x] for x in self.sep if x in content[0]}
            raw_values = content[1]
            self.cpa = cpa
            self.create_micro_agent(cpa, raw_values, init=False)

        elif typ == 'CPA':
            cpa, self.ub = content
            self.cpa = {x: cpa[x] for x in self.sep}
            self.init_vars()
            if self.level <= PTISBBAgent.ordering_level and not self.leaf_agent:
                self.create_micro_agent(cpa, True)
                self.micro_agent.active = True
            elif self.level <= PTISBBAgent.upper_bound_level and not self.leaf_agent:
                self.create_micro_agent(cpa, False)
                self.micro_agent.active = True
                self.cur_var = True
            else:
                self.continue_assign()
        elif typ == 'BACKTRACK':
            cur_idx = self.srch_val_idx[sender]
            cur_val = self.domain_vals[cur_idx]
            self.lb_c[sender][cur_val] = content
            self.backtrack_rcv_cnt[cur_val] += 1
            assert self.backtrack_rcv_cnt[cur_val] <= len(self.children)
            if self.backtrack_rcv_cnt[cur_val] == len(self.children):
                self.complete_vals.add(cur_val)
                self.ub = min(self.ub, self.lb(cur_val))
            cur_idx = self.next_feasible_val_idx(cur_idx)
            self.srch_val_idx[sender] = cur_idx
            if cur_idx != -1:
                cur_val = self.domain_vals[cur_idx]
                self.send_cpa(cur_val, sender)
            else:
                self.children_ready_cnt += 1
                if self.children_ready_cnt == len(self.children):
                    if self.root_agent:
                        print(f'Optimal cost: {self.ub}')
                        for c in self.children:
                            self.send_message(c, None, 'TERM')
                        self.stop()
                    else:
                        self.send_backtrack()
        elif typ == 'TERM':
            for c in self.children:
                self.send_message(c, None, 'TERM')
            self.stop()
        elif typ == 'COST_ACCUM':
            self.desc_cost += content
            self.desc_cnt += 1
            if self.desc_cnt == len(self.children):
                cost = self.desc_cost
                for p in self.all_parents:
                    oppo_val = self.cpa[p] if p in self.cpa else self.assign[p]
                    cost += self.constraint_functions[p][self.val][oppo_val]
                if self.cur_var:
                    if cost < self.ub:
                        self.ub = cost
                    self.continue_assign()
                else:
                    self.send_message(self.parent, cost, 'COST_ACCUM')
                self.desc_cost = self.desc_cnt = 0
                self.assign = None
                self.cur_var = False
        elif typ == 'ASSIGN':
            self.assign = dict(content)
            if self.micro_agent is None:
                self.dispose_assign()

        elif self.micro_agent is not None:
            self.micro_agent.dispose_msg(typ, sender, content)
        self.stop_watch.stop()

    def dispose_assign(self):
        assign = dict(self.assign)
        assign[self.id] = self.val
        for c in self.children:
            self.send_message(c, assign, 'ASSIGN')
        if self.leaf_agent:
            cost = 0
            for p in self.all_parents:
                oppo_val = self.cpa[p] if p in self.cpa else self.assign[p]
                cost += self.constraint_functions[p][self.val][oppo_val]
            self.send_message(self.parent, cost, 'COST_ACCUM')
            self.assign = None

    def continue_assign(self):
        if self.leaf_agent:
            self.send_backtrack()
        else:
            idx = self.next_feasible_val_idx(-1)
            if idx != -1:
                for c in self.children:
                    self.srch_val_idx[c] = idx
                    self.send_cpa(self.domain_vals[idx], c)
            else:
                if not self.root_agent:
                    self.send_backtrack()
                else:
                    print(f'Optimal cost: {self.ub}')
                    for c in self.children:
                        self.send_message(c, None, 'TERM')
                    self.stop()

    def create_micro_agent(self, cpa, raw_values, init=True):
        self.desc_cost = self.desc_cnt = 0
        self.assign = None
        if init:
            self.debug(f'{self.id} start gnn, {raw_values}')
        constraint_functions = dict()
        unary_function = [0] * self.domain
        all_parents = set()
        sep = {x for x in self.sep if x not in cpa}
        for n in self.neighbors:
            if n not in cpa:
                constraint_functions[n] = self.constraint_functions[n]
                if n in self.all_parents:
                    all_parents.add(n)
            else:
                uf = self.constraint_functions[n]
                uf = [uf[x][cpa[n]] for x in range(self.domain)]
                unary_function = [x + y for x, y in zip(uf, unary_function)]
        unary_function = [x / 100 for x in unary_function]
        all_layers = [PTISBBAgent.model.conv1, PTISBBAgent.model.conv2, PTISBBAgent.model.conv3,
                      PTISBBAgent.model.conv4]
        self.micro_agent = MicroAgent(all_layers, constraint_functions, self.domain,
                                      self.id, PTISBBAgent.model, self.neighbor_domain, self.on_micro_agent_stop,
                                      self.send_message, raw_values)
        self.micro_agent.unary_function = unary_function
        self.micro_agent.all_children.update(self.children)
        self.micro_agent.all_children.update(self.pseudo_children)
        self.micro_agent.children.update(self.children)
        self.micro_agent.all_parents = all_parents
        self.micro_agent.leaf_agent = len(self.children) == 0
        self.micro_agent.parent = None if len(self.micro_agent.all_parents) == 0 else self.parent
        self.micro_agent.sep = sep
        for c in self.children:
            self.send_message(c, (cpa, raw_values), 'BUILD_MICRO_AGENT')
        self.micro_agent.pseudo_tree_created()

    def send_cpa(self, val, child):
        cpa = dict(self.cpa)
        cpa[self.id] = val
        ub = self.ub - self.high_cost[val] - sum([self.lb_c[c][val] for c in self.children if c != child])
        self.send_message(child, (cpa, ub), 'CPA')

    def send_backtrack(self):
        if len(self.complete_vals) == 0:
            self.send_message(self.parent, MAX_VALUE, 'BACKTRACK')
        else:
            min_cost = MAX_VALUE
            for val in self.complete_vals:
                min_cost = min(min_cost, self.lb(val))
            self.send_message(self.parent, min_cost, 'BACKTRACK')

    def next_feasible_val_idx(self, prev_idx):
        found = False
        for idx in range(prev_idx + 1, self.domain):
            lb = self.lb(self.domain_vals[idx])
            if lb <= self.ub:
                if lb == self.ub:
                    if self.bound_enforced_val != -1:
                        continue
                    self.bound_enforced_val = self.domain_vals[idx]
                found = True
                break
        return idx if found else -1

    def lb(self, val):
        return self.high_cost[val] + sum([self.lb_c[c][val] for c in self.children])

    def init_vars(self):
        self.bound_enforced_val = -1
        self.cur_var = False
        self.children_ready_cnt = 0
        self.complete_vals = set()
        self.domain_vals = [i for i in range(self.domain)]
        if self.leaf_agent:
            self.complete_vals.update(self.domain_vals)
        self.high_cost = [0] * self.domain
        self.backtrack_rcv_cnt = [0] * self.domain
        for val in self.domain_vals:
            self.high_cost[val] = sum([self.constraint_functions[p][val][self.cpa[p]] for p in self.all_parents])
            self.ncccs += len(self.all_parents)
        self.srch_val_idx = dict()
        for c in self.children:
            self.srch_val_idx[c] = -1
            self.lb_c[c] = [0] * self.domain
            for val in self.domain_vals:
                cpa = dict(self.cpa)
                cpa[self.id] = val
                self.lb_c[c][val] = self.children_utils[c].reduce(cpa, eval=True)
        if PTISBBAgent.lb_ordering:
            dom_vals = [(x, self.lb(x)) for x in range(self.domain)]
            dom_vals = sorted(dom_vals, key=lambda x: x[-1])
            self.domain_vals = [x[0] for x in dom_vals]

    def send_sep_level(self):
        if len(self.sep_level) == len(self.sep) and self.pseudo_tree_created:
            ordering = sorted([(x, self.sep_level[x]) for x in self.sep], key=lambda x: x[1])
            self.sep_ordering = [x[0] for x in ordering]
            sep_level = dict(self.sep_level)
            sep_level[self.id] = self.level
            for c in self.children:
                self.send_message(c, sep_level, 'LEVEL')
            if self.leaf_agent:
                self.inference()

    def inference(self):
        buckets = [Bucket.from_matrix(self.constraint_functions[p], self.id, p) for p in self.all_parents]
        for bkt in self.children_utils.values():
            buckets.append(bkt)
        bucket = Bucket.join(buckets)
        bucket = bucket.proj(self.id)
        for bkt in self.children_utils.values():
            bkt.squeeze()
        self.ncccs += bucket.data.numel()
        drop_cnt = len(bucket.dims) - PTISBBAgent.k
        if drop_cnt > 0:
            cnt = 0
            for dim in self.sep_ordering:
                if dim in bucket.dims:
                    cnt += 1
                    bucket = bucket.proj(dim)
                    if cnt == drop_cnt:
                        break
        self.send_message(self.parent, bucket, 'UTIL')

    def send_message(self, dest, msg_content, msg_type=None):
        self._send_message(dest, msg_content, msg_type, statistics=(self.ncccs, self.stop_watch.elapse))

    def on_receive(self, msg, statistics):
        super(PTISBBAgent, self).on_receive(msg, statistics)
        if statistics is None:
            return
        ncccs, elapse = statistics
        self.ncccs = max(self.ncccs, ncccs)
        self.stop_watch.update(elapse)

    def stop(self):
        super(PTISBBAgent, self).stop()
        if self.mailer.result_obj is None:
            assert self.root_agent
            self.mailer.result_obj = (self.ncccs, self.stop_watch.elapse, self.ub)
        else:
            ncccs, elapse, ub = self.mailer.result_obj
            self.mailer.result_obj = (max(self.ncccs, ncccs), max(self.stop_watch.elapse, elapse), ub)
