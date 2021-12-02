import os

import torch
from torch_geometric.data import DataLoader
import torch.nn.functional as F

from .memory import Mem
from .primal_graph import PrimalGraph


class GATPCM:
    def __init__(self, env, model, x_init_feature, c_init_feature, f_init_feature, model_path, episode=100000, iteration=100000,
                 capacity=1000000, batch_size=32, validation=50, device='cpu'):
        self.episode = episode
        self.env = env
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
        self.x_init_feature = x_init_feature
        self.c_init_feature = c_init_feature
        self.f_init_feature = f_init_feature
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model_path = model_path
        self.batch_size = batch_size
        self.iteration = iteration
        self.validation = validation
        self.mem = Mem(capacity)

    def train(self):
        train_it = 0
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        for ep in range(self.episode):
            agents, root = self.env.reset(solve=True)
            primal_graph = PrimalGraph(agents, self.x_init_feature, self.c_init_feature, self.f_init_feature)
            non_leaf_vars = [a for a in agents.values() if not a.leaf_agent]
            idx = 1
            for n in non_leaf_vars:
                print(f'generating labelled data for {idx} / {len(non_leaf_vars)} variable')
                idx += 1
                for i in range(len(n.sep_assigns)):
                    sep_assign = {n.all_vars[j]: n.sep_assigns[i][j] for j in range(len(n.all_vars))}
                    cost = [n.optim[i, j].item() for j in range(n.domain)]
                    data = primal_graph.build_graph(n.id, sep_assign)
                    data.cost = cost
                    self.mem.add(data)
            for it in range(self.iteration):
                self.losses = []
                costs = []
                self._dfs_decision_making(agents, root, {}, primal_graph, costs)
                before_cost = sum(costs)

                for _ in range(1000):
                    self._learn()

                true_cost = root.optim.min().item()
                costs = []
                self._dfs_decision_making(agents, root, {}, primal_graph, costs)

                print(
                    f'Episode {ep} / {it} / {train_it}: {sum(self.losses) / len(self.losses): .4f}. True cost: {true_cost * 100: .1f}, before cost: {before_cost}, after cost: {sum(costs)}')

                if train_it % self.validation == 0:
                    cost = 0
                    for p in range(len(self.env.valid)):
                        test_agents, test_root = self.env.reset(p, solve=False)
                        test_primal_graph = PrimalGraph(test_agents, self.x_init_feature, self.c_init_feature, self.f_init_feature)
                        costs = []
                        self._dfs_decision_making(test_agents, test_root, {}, test_primal_graph, costs)
                        cost += sum(costs)
                    print(f'Validate {ep}: {cost / len(self.env.valid)}')
                    torch.save(self.model.state_dict(), f'{self.model_path}/{ep}_{it}_{train_it}.pth')
                train_it += 1

    def _dfs_decision_making(self, agents, cur_node, assign, primal_graph, costs):
        sep_assign = {x: assign[x] for x in cur_node.sep}
        if cur_node.leaf_agent:
            vec = [sum([cur_node.constraint_functions[p][val][assign[p]] for p in cur_node.all_parents]) for val in range(cur_node.domain)]
            costs.append(min(vec))
            return
        graph = primal_graph.build_graph(cur_node.id, sep_assign)
        values = self.model.inference(graph.x.to(self.device), graph.edge_index.to(self.device), graph.decision_var_idx, graph.function_idx)
        val = values.argmin().item()
        assign[cur_node.id] = val
        costs.append(sum([cur_node.constraint_functions[p][val][assign[p]] for p in cur_node.all_parents]))
        for c in cur_node.children:
            c = agents[c]
            self._dfs_decision_making(agents, c, assign, primal_graph, costs)

    def _learn(self):
        data, decision_vars, target = self.mem.sample(self.batch_size)
        self.optimizer.zero_grad()
        self.model.train()
        for b in DataLoader(data, batch_size=len(decision_vars)):
            b.x = b.x.to(self.device)
            b.edge_index = b.edge_index.to(self.device)
            b.batch = b.batch.to(self.device)
            pred = self.model(b, decision_vars)
        target = torch.tensor(target, device=self.device).view((-1, 1))
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        self.losses.append(loss.item())
        self.model.eval()