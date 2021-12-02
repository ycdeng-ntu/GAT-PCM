import ast

import torch
from torch_geometric.data import Data


class PrimalGraph:
    def __init__(self, agents, x_feature, c_feature, f_feature):
        self.agents = agents
        self.x_feature = x_feature
        self.c_feature = c_feature
        self.f_feature = f_feature

    def build_graph(self, root, sep_assign):
        x = []
        edge_index = [[], []]
        decision_var_idx = dict()
        self._dfs_var(root, x, decision_var_idx)
        function_idx = []
        self._dfs_constraint(root, x, decision_var_idx, edge_index, sep_assign, function_idx)
        return Data(x=torch.tensor(x), edge_index=torch.tensor(edge_index), decision_var_idx=decision_var_idx[root], function_idx=function_idx)

    def _dfs_var(self, agent_id, x, decision_var_idx):
        agent = self.agents[agent_id]
        decision_var_idx[agent_id] = []
        for d in range(agent.domain):
            decision_var_idx[agent_id].append(len(x))
            x.append(list(self.x_feature))
        for c in agent.children:
            self._dfs_var(c, x, decision_var_idx)

    def _dfs_constraint(self, agent_id, x, decision_var_idx, edge_index, sep_assign, function_idx):
        src, dest = edge_index
        agent = self.agents[agent_id]
        fixed_parents = {i for i in agent.all_parents if i in sep_assign}
        fixed_costs = [sum([agent.constraint_functions[i][val][sep_assign[i]] for i in fixed_parents]) / 100 for val in range(agent.domain)]
        if len(fixed_parents) != 0:
            cost_node_idx = []
            for my_val in range(agent.domain):
                m_idx = decision_var_idx[agent_id][my_val]
                c_idx = len(x)
                cost_node_idx.append(c_idx)
                src += [c_idx]
                dest += [m_idx]
                x.append(ast.literal_eval(self.c_feature.format(fixed_costs[my_val])))
            for c_idx in cost_node_idx:
                src.append(c_idx)
                dest.append(len(x))
            function_idx.append(len(x))
            x.append(self.f_feature)

        for p in agent.all_parents:
            if p in fixed_parents:
                continue
            cost_node_idx = []
            matrix = agent.constraint_functions[p]
            for my_val in range(agent.domain):
                m_idx = decision_var_idx[agent_id][my_val]
                for your_val in range(self.agents[p].domain):
                    y_idx = decision_var_idx[p][your_val]
                    c_idx = len(x)
                    cost_node_idx.append(c_idx)
                    src += [m_idx, c_idx]
                    dest += [c_idx, y_idx]
                    x.append(ast.literal_eval(self.c_feature.format(matrix[my_val][your_val] / 100)))

            for c_idx in cost_node_idx:
                src.append(c_idx)
                dest.append(len(x))
            function_idx.append(len(x))
            x.append(self.f_feature)
        for c in agent.children:
            self._dfs_constraint(c, x, decision_var_idx, edge_index, sep_assign, function_idx)