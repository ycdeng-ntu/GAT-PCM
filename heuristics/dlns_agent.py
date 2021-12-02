import random

from core.agent import Agent
from core.utilities import argmin
from .micro_agent import MicroAgent


class DLNSAgent(Agent):
    p = 0.2
    model = None
    cycle_cnt = 1000

    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.cycle = 0
        self.destroyed = False
        self.destroyed_view = dict()
        self.micro_agent = None
        self.ready_cnt = 0
        self.eval_local_cost = True
        self.traffic = 0
        self.traffic_in_cycle = []

    def start(self):
        self.val = random.randrange(self.domain)
        self.start_dlns()

    def start_dlns(self):
        self.destroyed_view.clear()
        self.destroyed = random.random() < DLNSAgent.p
        self.ready_cnt = 0
        self.micro_agent = None
        for n in self.neighbors:
            self.send_message(n, (self.destroyed, self.val), 'DES')

    def tell_root_ready(self):
        self.send_message(self.mailer.root, None, 'READY')

    def reset_topology(self):
        self.micro_agent.children.clear()
        self.micro_agent.all_children.clear()
        self.micro_agent.all_parents.clear()
        self.micro_agent.sep.clear()
        self.micro_agent.active = False
        self.micro_agent.leaf_agent = False
        self.micro_agent.parent = -1

    def send_message(self, dest, msg_content, msg_type=None):
        if msg_type is not None and type(msg_content) != type(None):
            if msg_type == 'DES':
                self.traffic += 2
            elif msg_type == 'DFS':
                self.traffic += len(msg_content[0]) + 1
            elif msg_type == 'DFS_BAK':
                self.traffic += len(msg_content[0]) + len(msg_content[1]) + 1
            elif msg_type == 'UPDATE':
                self.traffic += msg_content[0].numel()
            elif msg_type == 'AGG':
                self.traffic += msg_content.numel()
            elif msg_type == 'VAL':
                self.traffic += len(msg_content[0]) + 1
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
            if len(self.destroyed_view) == len(self.neighbors):
                self.traffic_in_cycle.append(self.traffic)
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
                    if len(free_neighbors) == 0:
                        self.val = argmin([sum([self.constraint_functions[n][val_][self.local_view[n]] for n in
                                                self.neighbors]) for val_ in range(self.domain)])
                        self.destroyed = False
                        self.tell_root_ready()
                        self.debug(f'{self.id} repaired')
                    else:
                        # build micro agent & start dfs
                        all_layers = [DLNSAgent.model.conv1, DLNSAgent.model.conv2, DLNSAgent.model.conv3,
                                      DLNSAgent.model.conv4]
                        cf = dict()
                        for n in free_neighbors:
                            m = []
                            for row in self.constraint_functions[n]:
                                m.append([x for x in row])
                            cf[n] = m
                        self.micro_agent = MicroAgent(all_layers, cf, self.domain, self.id, DLNSAgent.model,
                                                      self.neighbor_domain, self.on_micro_agent_stop, self.send_message)
                        unary_func = [
                            sum([self.constraint_functions[n][val_][self.local_view[n]] for n in self.neighbors if
                                 not self.destroyed_view[n]]) / 100 for val_ in range(self.domain)]
                        if sum(unary_func) != 0:
                            self.micro_agent.unary_function = unary_func

                        child = random.choice(list(self.micro_agent.constraint_functions.keys()))
                        self.micro_agent.children.add(child)
                        self.micro_agent.all_children.add(child)
                        visited = set()
                        visited.add(self.id)
                        self.micro_agent.root_id = self.id
                        self.debug(f'{self.id} start dfs')
                        self.send_message(child, (visited, self.id), 'DFS')
        elif typ == 'DFS':
            visited, root_id = content
            assert self.micro_agent is not None
            if root_id < self.micro_agent.root_id:
                self.debug(f'{self.id} aborts dfs {self.micro_agent.root_id} because {root_id}')
                self.reset_topology()
                self.micro_agent.root_id = root_id
            if root_id == self.micro_agent.root_id:
                assert root_id != self.id
                self.micro_agent.parent = sender
                visited.add(self.id)
                next_agent = ''
                neighbors = list(self.micro_agent.constraint_functions.keys())
                random.shuffle(neighbors)
                for n in neighbors:
                    if n in visited:
                        self.micro_agent.all_parents.add(n)
                    else:
                        next_agent = n
                if next_agent != '':
                    self.micro_agent.children.add(next_agent)
                    self.micro_agent.all_children.add(next_agent)
                    self.send_message(next_agent, (visited, self.micro_agent.root_id), 'DFS')
                else:
                    self.micro_agent.leaf_agent = True
                    self.micro_agent.sep = set(self.micro_agent.all_parents)
                    self.send_message(self.micro_agent.parent, (visited, self.micro_agent.sep, self.micro_agent.root_id), 'DFS_BAK')
        elif typ == 'DFS_BAK':
            visited, sep, root_id = content
            if root_id != self.micro_agent.root_id:
                self.debug(f'{root_id} dfs is aborted because {self.micro_agent.root_id}')
                assert root_id > self.micro_agent.root_id
            else:
                self.micro_agent.sep.update(sep)
                next_agent = ''
                neighbors = list(self.micro_agent.constraint_functions.keys())
                random.shuffle(neighbors)
                for n in neighbors:
                    if n in visited and n not in self.micro_agent.all_parents:
                        self.micro_agent.all_children.add(n)
                    if n not in visited:
                        next_agent = n
                if next_agent != '':
                    self.micro_agent.children.add(next_agent)
                    self.micro_agent.all_children.add(next_agent)
                    self.send_message(next_agent, (visited, self.micro_agent.root_id), 'DFS')
                else:
                    self.micro_agent.sep.update(self.micro_agent.all_parents)
                    self.micro_agent.sep.discard(self.id)
                    if self.micro_agent.root_id != self.id:
                        self.send_message(self.micro_agent.parent,
                                           (visited, self.micro_agent.sep, self.micro_agent.root_id), 'DFS_BAK')
                    else:
                        assert len(self.micro_agent.all_parents) == len(self.micro_agent.sep) == 0
                        self.micro_agent.active = True
                        for child in self.micro_agent.children:
                            self.send_message(child, None, 'DFS_READY')
                        self.debug(f'id: {self.id}, children: {self.micro_agent.children}, all_children: {self.micro_agent.all_children}, parent: {self.micro_agent.parent}, all_parent: {self.micro_agent.all_parents}, sep: {self.micro_agent.sep}')
                        self.micro_agent.pseudo_tree_created()
        elif typ == 'DFS_READY':
            for child in self.micro_agent.children:
                self.send_message(child, None, 'DFS_READY')
            self.debug(
                f'id: {self.id}, children: {self.micro_agent.children}, all_children: {self.micro_agent.all_children}, parent: {self.micro_agent.parent}, all_parent: {self.micro_agent.all_parents}, sep: {self.micro_agent.sep}')
            self.micro_agent.pseudo_tree_created()
        elif typ == 'READY':
            self.ready_cnt += 1
            if self.ready_cnt == self.mailer.agent_cnt:
                for a in self.mailer.agents.keys():
                    self.send_message(a, None, 'NEW_CYCLE')
        elif typ == 'NEW_CYCLE':
            self.start_dlns()
        else:
            assert self.micro_agent is not None
            self.micro_agent.dispose_msg(typ, sender, content)
        self.stop_watch.stop()

    def on_micro_agent_stop(self):
        self.val = self.micro_agent.val
        self.debug(f'{self.id}: {self.val}')
        self.tell_root_ready()

    def stop(self):
        super(DLNSAgent, self).stop()
        if self.mailer.result_obj is None:
            self.mailer.result_obj = (self.stop_watch.elapse, list(self.traffic_in_cycle))
        else:
            elapse, traffic_in_cycle = self.mailer.result_obj
            self.mailer.result_obj = (max(self.stop_watch.elapse, elapse), [x + y for x, y in zip(traffic_in_cycle, self.traffic_in_cycle)])