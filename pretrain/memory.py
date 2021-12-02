import random

from torch_geometric.data import Data


class Mem:
    def __init__(self, capacity):
        self.data = [0] * capacity
        self.capacity = capacity
        self.size = 0
        self.pointer = 0

    def add(self, data):
        self.data[self.pointer] = data
        self.pointer += 1
        self.pointer %= self.capacity
        self.size += 1
        self.size = min(self.size, self.capacity)

    def sample(self, n):
        batch_size = n
        n = min(n, self.size)
        idxes = random.sample(range(self.size), n)
        data = []
        decision_vars = []
        targets = []
        for idx in idxes:
            d = self.data[idx]
            for i in range(len(d.decision_var_idx)):
                decision_vars.append(d.decision_var_idx[i])
                targets.append(d.cost[i])
                da = Data(x=d.x, edge_index=d.edge_index, function_idx=d.function_idx)
                data.append(da)
                if len(data) == batch_size:
                    break
            if len(data) == batch_size:
                break
        return data, decision_vars, targets