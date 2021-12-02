import ast
import torch.nn.functional as F

import torch

from core.utilities import argmin

x_init_feat = [0, 0, 1, 0]
c_init_feat = '[0, 1, 0, {}]'
f_init_feat = [1, 0, 0, 0]
device = torch.device('cpu')

def check():
    print(device)


def build_init_embedding(self, sep_assign):
    self.all_rcv_embeddings.clear()
    for i in range(len(self.all_layers) - 1):
        self.all_rcv_embeddings.append(dict())
    self.embedding = []
    self.edge_index = []
    self.function_idx.clear()
    self.update_to.clear()
    self.rcv_from.clear()
    self.msg_rcv_cnt.clear()
    self.msg_rcv_cnt['UPDATE'] = 0
    self.agg_func_embedding = None
    self.ready = False
    for val in range(self.domain):
        self.embedding.append(x_init_feat)
    fixed_parents = {i for i in self.all_parents if i in sep_assign}
    fixed_costs = [sum([self.constraint_functions[i][val][sep_assign[i]] for i in fixed_parents]) / 100 for val in
                   range(self.domain)]
    src = []
    dest = []
    self.edge_index = [src, dest]
    if len(fixed_parents) != 0 or hasattr(self, 'unary_function'):
        cost_node_idx = []
        if hasattr(self, 'unary_function'):
            fixed_costs = [x + y for x, y in zip(self.unary_function, fixed_costs)]
        for my_val in range(self.domain):
            m_idx = my_val
            c_idx = len(self.embedding)
            cost_node_idx.append(c_idx)
            src += [c_idx]
            dest += [m_idx]
            self.embedding.append(ast.literal_eval(c_init_feat.format(fixed_costs[my_val])))
        for c_idx in cost_node_idx:
            src.append(c_idx)
            dest.append(len(self.embedding))
        self.function_idx.append(len(self.embedding))
        self.embedding.append(f_init_feat)
    for p in self.all_parents:
        if p in fixed_parents:
            continue
        cost_node_idx = []
        matrix = self.constraint_functions[p]
        for my_val in range(self.domain):
            m_idx = my_val
            for your_val in range(self.neighbor_domain(p)):
                c_idx = len(self.embedding)
                cost_node_idx.append(c_idx)
                src += [m_idx]
                dest += [c_idx]
                self.embedding.append(ast.literal_eval(c_init_feat.format(matrix[my_val][your_val] / 100)))
        self.update_to[p] = cost_node_idx
        for c_idx in cost_node_idx:
            src.append(c_idx)
            dest.append(len(self.embedding))
        self.function_idx.append(len(self.embedding))
        self.embedding.append(f_init_feat)
    for c in self.all_children:
        cost_node_idx = []
        matrix = self.constraint_functions[c]
        for your_val in range(self.neighbor_domain(c)):
            for my_val in range(self.domain):
                c_idx = len(self.embedding)
                cost_node_idx.append(c_idx)
                src += [c_idx]
                dest += [my_val]
                self.embedding.append(
                    ast.literal_eval(c_init_feat.format(matrix[my_val][your_val] / 100)))
        self.rcv_from[c] = cost_node_idx
    self.embedding = torch.tensor(self.embedding).to(device)
    self.edge_index = torch.tensor(self.edge_index).to(device)


def send_update(self):
    for p in self.update_to.keys():
        self._send_message(p, (self.embedding[self.update_to[p]], self.layer_idx), 'UPDATE')
    if self.leaf_agent:
        self._send_message(self.id, None, 'UPDATE')


def agg_and_making_decision(self, model):
    if self.msg_rcv_cnt['AGG'] == len(self.children) and self.ready:
        if not self.active:
            self._send_message(self.parent, self.agg_func_embedding, 'AGG')
            if self.raw_values:
                self.stop()
        else:
            self.agg_func_embedding = model.agg(self.agg_func_embedding)
            agg = self.agg_func_embedding.repeat((self.domain, 1))
            trans = self.embedding[:self.domain, :]
            trans = model.transform(trans)
            values = model.out(F.elu(torch.cat([agg, trans], dim=1)))
            if self.raw_values:
                self.values = values.squeeze()
                self.stop()
                return
            self.val = values.argmin().item()
            sep_assign = dict(self.sep_assign)
            sep_assign[self.id] = self.val
            self.active = False
            for c in self.children:
                self._send_message(c, (sep_assign, self.id), 'VAL')
            self.partial_cost = sum([self.constraint_functions[p][self.val][sep_assign[p]] for p in self.all_parents])
            self.stop()
        self.ready = False
        self.msg_rcv_cnt.pop('AGG')


def dispose_update_msg(self, sender, content, model):
    if not self.leaf_agent:
        content, timestamp = content
        self.all_rcv_embeddings[timestamp - 1][sender] = content
    if len(self.all_rcv_embeddings[self.layer_idx - 1]) == len(self.rcv_from):
        for sender, content in self.all_rcv_embeddings[self.layer_idx - 1].items():
            self.embedding[self.rcv_from[sender]] = content
        if self.layer_idx < len(self.all_layers):
            self.embedding = F.elu(self.all_layers[self.layer_idx](self.embedding, self.edge_index))
            self.layer_idx += 1
            if self.layer_idx < len(self.all_layers):
                send_update(self)
            else:
                assert self.layer_idx == len(self.all_layers)
                self.ready = True
                if 'AGG' not in self.msg_rcv_cnt:
                    self.agg_func_embedding = torch.sum(self.embedding[self.function_idx], dim=0)
                    self.msg_rcv_cnt['AGG'] = 0
                else:
                    self.agg_func_embedding = self.agg_func_embedding + torch.sum(self.embedding[self.function_idx],
                                                                                  dim=0)
                    agg_and_making_decision(self, model)
                if self.leaf_agent:
                    self._send_message(self.parent, self.agg_func_embedding, 'AGG')
                    if self.raw_values:
                        self.stop()


def dispose_agg_msg(self, content, model):
    if 'AGG' not in self.msg_rcv_cnt:
        self.agg_func_embedding = torch.zeros(content.shape[0], device=device)
        self.msg_rcv_cnt['AGG'] = 0
    self.agg_func_embedding = self.agg_func_embedding + content
    self.msg_rcv_cnt['AGG'] += 1
    agg_and_making_decision(self, model)


def dispose_val_msg(self, content):
    sep_assign, assigned_id = content
    if assigned_id == self.parent:
        self.active = True
    self.sep_assign = {x: sep_assign[x] for x in self.sep if x in sep_assign}
    if self.leaf_agent and self.active:
        unary_function = [sum([self.constraint_functions[p][val][sep_assign[p]] for p in self.constraint_functions.keys()]) / 100 for val in
                           range(self.domain)]
        if hasattr(self, 'unary_function'):
            for i in range(self.domain):
                unary_function[i] += self.unary_function[i]
        self.val = argmin(unary_function)
        self.partial_cost = sum(
            [self.constraint_functions[p][self.val][sep_assign[p]] for p in self.all_parents])
        self.stop()
        return
    build_init_embedding(self, self.sep_assign)
    self.embedding = F.elu(self.all_layers[0](self.embedding.double(), self.edge_index.long()))
    self.layer_idx = 1
    self.msg_rcv_cnt['UPDATE'] = 0
    send_update(self)
    for c in self.children:
        self._send_message(c, (sep_assign, assigned_id), 'VAL')