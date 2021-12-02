import random

from core.agent import Agent
from core.utilities import argmin


class DTree:
    def __init__(self, id, domain, send_message, on_stop, unary_costs=None):
        self.id = id
        self.domain = domain
        self.send_message = send_message
        self.unary_costs = unary_costs
        self.on_stop = on_stop
        self.parent = None
        self.children = None
        self.cost_table = list()
        self.util_rcv_cnt = 0
        self.children_util = [0] * domain
        self.val = 0

    def on_tree_created(self, parent, cost_matrix, children):
        if self.unary_costs is None:
            self.unary_costs = [0] * self.domain
        if parent is not None:
            self.parent = parent
            for val in range(self.domain):
                costs = [x + self.unary_costs[val] for x in cost_matrix[val]]
                self.cost_table.append(costs)
        self.children = set(children)

        if len(self.children) == 0:
            proj = []
            for val1 in range(len(self.cost_table[0])):
                best_cost = self.cost_table[0][val1]
                for val in range(self.domain):
                    best_cost = min(best_cost, self.cost_table[val][val1])
                proj.append(best_cost)
            self.send_message(self.parent, proj, 'UTIL')

    def dispose_msg(self, typ, sender, content):
        if typ == 'UTIL':
            self.util_rcv_cnt += 1
            self.children_util = [x + y for x, y in zip(content, self.children_util)]
            if self.util_rcv_cnt == len(self.children):
                if self.parent is not None:
                    for val in range(self.domain):
                        for val1 in range(len(self.cost_table[0])):
                            self.cost_table[val][val1] += self.children_util[val]
                    proj = []
                    for val1 in range(len(self.cost_table[0])):
                        best_cost = self.cost_table[0][val1]
                        for val in range(self.domain):
                            best_cost = min(best_cost, self.cost_table[val][val1])
                        proj.append(best_cost)
                    self.send_message(self.parent, proj, 'UTIL')
                else:
                    self.val = argmin([x + y for x, y in zip(self.unary_costs, self.children_util)])
                    for c in self.children:
                        self.send_message(c, self.val, 'VAL')
                    self.on_stop()
        elif typ == 'VAL':
            self.val = argmin([self.cost_table[i][content] for i in range(self.domain)])
            for c in self.children:
                self.send_message(c, self.val, 'VAL')
            self.on_stop()


class DLNSAgent(Agent):
    p = 0.5
    cycle_cnt = 10000

    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.cycle = 0
        self.destroyed = False
        self.destroyed_view = dict()
        self.dtree = None
        self.ready_cnt = 0
        self.eval_local_cost = True
        self.traffic = 0
        self.parent = None
        self.children = set()
        self.free_neighbors = list()
        self.root_id = None
        self.traffic_in_cycle = []

    def start(self):
        self.val = random.randrange(self.domain)
        self.start_dlns()

    def start_dlns(self):
        self.destroyed_view.clear()
        self.destroyed = random.random() < DLNSAgent.p
        self.ready_cnt = 0
        self.dtree = None
        self.free_neighbors = list()
        self.reset_topology()
        for n in self.neighbors:
            self.send_message(n, (self.destroyed, self.val), 'DES')

    def tell_root_ready(self):
        self.send_message(self.mailer.root, None, 'READY')

    def reset_topology(self):
        self.parent = None
        self.children = set()
        self.root_id = None

    def send_message(self, dest, msg_content, msg_type=None):
        if msg_type is not None and type(msg_content) != type(None):
            if msg_type == 'DES':
                self.traffic += 2
            elif msg_type == 'DFS':
                self.traffic += len(msg_content[0]) + 1
            elif msg_type == 'UTIL':
                self.traffic += len(msg_content)
            elif msg_type == 'VAL':
                self.traffic += 1
        self._send_message(dest, msg_content, msg_type, statistics=self.stop_watch.elapse)

    def on_receive(self, msg, statistics):
        super(DLNSAgent, self).on_receive(msg, statistics)
        self.stop_watch.update(statistics)

    def _dispose_message(self, msg):
        typ, sender, content = msg.type, msg.sender, msg.content
        self.stop_watch.start()
        if typ == 'DES':
            des, val = content
            self.local_view[sender] = val
            self.destroyed_view[sender] = des
            self.round_end = True
            self.traffic_in_cycle.append(self.traffic)
            if len(self.destroyed_view) == len(self.neighbors):
                self.cycle += 1
                if self.cycle == DLNSAgent.cycle_cnt:
                    self.stop()
                    self.stop_watch.stop()
                    return
                if not self.destroyed:
                    self.tell_root_ready()
                else:
                    self.debug(f'{self.id} destroyed')
                    free_neighbors = [x for x in self.neighbors if self.destroyed_view[x]]
                    self.free_neighbors = list(free_neighbors)
                    if len(free_neighbors) == 0:
                        self.val = argmin([sum([self.constraint_functions[n][val_][self.local_view[n]] for n in
                                                self.neighbors]) for val_ in range(self.domain)])
                        self.destroyed = False
                        self.tell_root_ready()
                        self.debug(f'{self.id} repaired')
                    else:
                        # build micro agent & start dfs
                        cf = dict()
                        for n in free_neighbors:
                            m = []
                            for row in self.constraint_functions[n]:
                                m.append([x for x in row])
                            cf[n] = m
                        unary_func = [
                            sum([self.constraint_functions[n][val_][self.local_view[n]] for n in self.neighbors if
                                 not self.destroyed_view[n]]) for val_ in range(self.domain)]
                        self.dtree = DTree(self.id, self.domain, self.send_message, self.on_micro_agent_stop, unary_func)

                        child = random.choice(self.free_neighbors)
                        self.children.add(child)
                        visited = set()
                        visited.add(self.id)
                        self.root_id = self.id
                        self.debug(f'{self.id} start dfs')
                        self.send_message(child, (visited, self.id), 'DFS')
        elif typ == 'DFS':
            visited, root_id = content
            assert self.dtree is not None
            if root_id < self.root_id:
                self.debug(f'{self.id} aborts dfs {self.root_id} because {root_id}')
                self.reset_topology()
                self.root_id = root_id
            if root_id == self.root_id:
                assert root_id != self.id
                self.parent = sender
                visited.add(self.id)
                next_agent = ''
                neighbors = list(self.free_neighbors)
                random.shuffle(neighbors)
                for n in neighbors:
                    if n not in visited:
                        next_agent = n
                        break

                if next_agent != '':
                    self.children.add(next_agent)
                    self.send_message(next_agent, (visited, self.root_id), 'DFS')
                else:
                    self.send_message(self.parent, (visited, self.root_id), 'DFS_BAK')
        elif typ == 'DFS_BAK':
            visited, root_id = content
            if root_id != self.root_id:
                self.debug(f'{root_id} dfs is aborted because {self.root_id}')
                assert root_id > self.root_id
            else:
                next_agent = ''
                neighbors = list(self.free_neighbors)
                random.shuffle(neighbors)
                for n in neighbors:
                    if n not in visited:
                        next_agent = n
                        break

                if next_agent != '':
                    self.children.add(next_agent)
                    self.send_message(next_agent, (visited, self.root_id), 'DFS')
                else:
                    if self.root_id != self.id:
                        self.send_message(self.parent, (visited, self.root_id), 'DFS_BAK')
                    else:
                        for child in self.children:
                            self.send_message(child, None, 'DFS_READY')
                        self.debug(f'id: {self.id}, children: {self.children}, parent: {self.parent}')
                        self.dtree.on_tree_created(None, None, self.children)
        elif typ == 'DFS_READY':
            for child in self.children:
                self.send_message(child, None, 'DFS_READY')
            self.debug(f'id: {self.id}, children: {self.children}, parent: {self.parent}')
            self.dtree.on_tree_created(self.parent, self.constraint_functions[self.parent], self.children)
        elif typ == 'READY':
            self.ready_cnt += 1
            if self.ready_cnt == self.mailer.agent_cnt:
                for a in self.mailer.agents.keys():
                    self.send_message(a, None, 'NEW_CYCLE')
        elif typ == 'NEW_CYCLE':
            self.start_dlns()
        else:
            assert self.dtree is not None
            self.dtree.dispose_msg(typ, sender, content)
        self.stop_watch.stop()

    def on_micro_agent_stop(self):
        self.val = self.dtree.val
        self.debug(f'{self.id}: {self.val}')
        self.tell_root_ready()

    def stop(self):
        super(DLNSAgent, self).stop()
        if self.mailer.result_obj is None:
            self.mailer.result_obj = (self.traffic_in_cycle, self.stop_watch.elapse)
        else:
            tic, elapse = self.mailer.result_obj
            self.mailer.result_obj = ([x + y for x, y in zip(tic, self.traffic_in_cycle)], max(self.stop_watch.elapse, elapse))