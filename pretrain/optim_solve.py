import itertools

import torch

from core.agent import DFSAgent
from core.bucket import Bucket


class OptimAgent(DFSAgent):
    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.sep_assigns = []
        self.optim = None
        self.buckets = []
        self.all_vars = []
        self.solve = True

    def _on_pseudo_tree_created(self):
        if not self.solve:
            self.terminate = True
            return
        if self.leaf_agent:
            buckets = []
            for p in self.all_parents:
                buckets.append(Bucket.from_matrix(self.constraint_functions[p], self.id, p))
            self.bucket = Bucket.join(buckets)
            self._send_message(self.parent, self.bucket.proj(self.id), 'UTIL')
            self.get_optim()
            self.terminate = True


    def _dispose_message(self, msg):
        super(OptimAgent, self)._dispose_message(msg)
        typ, content = msg.type, msg.content
        if typ == 'UTIL':
            self.buckets.append(content)
            if len(self.buckets) == len(self.children):
                self.buckets = self.buckets + [Bucket.from_matrix(self.constraint_functions[p], self.id, p) for p in
                                               self.all_parents]
                self.bucket = Bucket.join(self.buckets)
                if not self.root_agent:
                    self._send_message(self.parent, self.bucket.proj(self.id), 'UTIL')
                    self.get_optim()

                else:
                    # print(self.bucket.data.min().item())
                    self.sep_assigns.append(())
                    self.optim = self.bucket.data.view((-1, self.domain)) / 100
                self.terminate = True

    def get_optim(self):
        all_vars = list(self.sep) + [self.id]
        self.all_vars = all_vars[:-1]
        doms = [[i for i in range(self.domain)] for _ in range(len(all_vars))]
        self.bucket.align_(all_vars)
        assign = []
        for it in itertools.product(*doms):
            assign.append(it)
        doms = doms[:-1]
        for it in itertools.product(*doms):
            self.sep_assigns.append(it)
        assign = torch.LongTensor(assign)
        chunks = assign.chunk(chunks=len(all_vars), dim=1)
        self.optim = self.bucket.data[tuple(chunks)].view((-1, self.domain)) / 100
