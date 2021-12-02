import random

from core.agent_manager import AgentManager
from .optim_solve import OptimAgent


class Environment:
    def __init__(self, problem_paths, roots=None, train_valid_split=0.8):
        self.train = random.sample(problem_paths, int(train_valid_split * len(problem_paths)))
        self.valid = []
        for p in problem_paths:
            if p not in self.train:
                self.valid.append(p)

        self.roots = roots

    def reset(self, idx=-1, root=None, solve=True):
        pool = self.train if solve else self.valid
        if idx == -1:
            idx = random.randrange(0, len(pool))
        if root is None:
            root = random.choice(self.roots[pool[idx]]) if self.roots is not None else None
        am = AgentManager(pool[idx], OptimAgent, root=root, rnd_root=True)
        for a in am.agents.values():
            a.solve = solve
        am.run()
        agents = dict(am.agents)
        root_agent = agents[am.root]
        return agents, root_agent