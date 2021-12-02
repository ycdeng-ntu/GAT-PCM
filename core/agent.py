from queue import Queue, PriorityQueue
from enum import Enum, unique
from time import perf_counter

@unique
class MsgType(Enum):
    MSG_DFS = 0x223646
    MSG_BACKTRACK = 0X361245
    MSG_DEGREE = 0X123456
    MSG_READY = 0x192983


class StopWatch:
    def __init__(self):
        self.elapse = 0
        self.cur_t = -1

    def reset(self):
        self.elapse = 0
        self.cur_t = -1

    def start(self):
        self.cur_t = perf_counter()

    def stop(self):
        if self.cur_t < 0:
            return
        self.elapse += perf_counter() - self.cur_t
        self.cur_t = -1

    def update(self, another_elapse):
        self.elapse = max(self.elapse, another_elapse)


class Agent:
    tracer_id = None
    on_round_ends = None
    constraint_cnt = 0

    def __init__(self, id, domain, neighbors, constraint_functions):
        self.id = id
        self.domain = domain
        self.neighbors = neighbors
        self.constraint_functions = constraint_functions
        self.message_queue = Queue()
        self.terminate = False
        self.local_view = {}
        self.val = 0
        self.mailer = None
        self.eval_local_cost = False
        self.print_log = False
        self.post_execution = []
        self.stop_watch = StopWatch()
        self.round_end = True

    def get_local_cost(self):
        total_cost = 0
        for neighbor in self.neighbors:
            neighborVal = 0 if neighbor not in self.local_view else self.local_view[neighbor]
            total_cost += self.constraint_functions[neighbor][self.val][neighborVal]
        return total_cost / 2

    def start(self):
        pass

    def post(self, delay, func, args):
        self.post_execution.append([delay + 1, func, args])

    def timestep_advance(self):
        while not self.message_queue.empty():
            msg = self.message_queue.get()
            self._dispose_message(msg)
        local_cost = self.get_local_cost() if self.eval_local_cost else 0
        round_end = self.round_end
        if self.round_end:
            self.round_end = False
        if self.terminate:
            return local_cost, round_end

        timed_outs = []
        remaining = []
        for i in range(len(self.post_execution)):
            item = self.post_execution[i]
            item[0] -= 1
            if item[0] == 0:
                timed_outs.append((item[1], item[2]))
            else:
                remaining.append(item)
        self.post_execution = remaining
        for fun, args in timed_outs:
            fun(*args)
        self._on_timestep_advanced()
        return local_cost, round_end

    def _on_timestep_advanced(self):
        pass

    def _dispose_message(self, msg):
        pass

    def _on_checkpoint_fired(self):
        pass

    def __repr__(self):
        return self.id

    def neighbor_domain(self, neighbor):
        return len(self.constraint_functions[neighbor][0])

    def _send_message(self, dest, msg_content, msg_type=None, statistics=None):
        self.mailer.post_message(dest, Message(self.id, msg_content, msg_type), statistics)

    def on_receive(self, msg, statistics):
        self.message_queue.put(msg)

    def __lt__(self, other):
        return 0

    def stop(self):
        self.terminate = True

    def debug(self, s):
        if self.print_log:
            print(s)


class DFSAgent(Agent):
    total_height = 0

    def __init__(self, id, domain, neighbors, constraint_functions):
        super().__init__(id, domain, neighbors, constraint_functions)
        self.parent = -1
        self.pseudo_parents = set()
        self.children = set()
        self.pseudo_children = set()
        self.sep = set()
        self.degree_view = PriorityQueue()
        self.preprocessing = True
        self.pseudo_tree_created = False
        self.height = 0
        self.all_parents = set()
        self.trimmed_sep = set()
        self.level = 0


    def start(self):
        for neighbor in self.neighbors:
            self.mailer.post_message(neighbor, Message(self.id, len(self.neighbors), MsgType.MSG_DEGREE))

    def _dispose_message(self, msg):
        if msg.type == MsgType.MSG_DEGREE:
            num_id = int(msg.sender[1:])
            self.degree_view.put((-msg.content, num_id, msg.sender))
        elif msg.type == MsgType.MSG_DFS:
            self.parent = msg.sender
            visited = msg.content
            visited.add(self.id)
            next_agent = -1
            for neighbor in self.neighbors:
                if neighbor != self.parent and neighbor in visited:
                    self.pseudo_parents.add(neighbor)
            while not self.degree_view.empty():
                poped = self.degree_view.get()
                next_agent = poped[-1]
                if next_agent not in visited:
                    break
                else:
                    next_agent = -1
            if next_agent != -1:
                self.children.add(next_agent)
                self.mailer.post_message(next_agent, Message(self.id, visited, MsgType.MSG_DFS))
            else:
                self.height = 0
                self.sep = set(self.pseudo_parents)
                self.sep.add(self.parent)
                self.mailer.post_message(self.parent, Message(self.id, (visited, self.height + 1, set(self.sep)), MsgType.MSG_BACKTRACK))
        elif msg.type == MsgType.MSG_BACKTRACK:
            if self.print_log:
                print(self.id, 'ready')
            visited, height, sep = msg.content
            self.height = max(self.height, height)
            self.sep.update(sep)
            self.trimmed_sep.update(sep)
            for neighbor in self.neighbors:
                if neighbor in visited and neighbor != self.parent and neighbor not in self.pseudo_parents and neighbor not in self.children:
                    self.pseudo_children.add(neighbor)
            next_agent = -1
            while not self.degree_view.empty():
                next_agent = self.degree_view.get()[-1]
                if next_agent not in visited:
                    break
                else:
                    next_agent = -1
            if next_agent != -1:
                self.children.add(next_agent)
                self.mailer.post_message(next_agent, Message(self.id, visited, MsgType.MSG_DFS))
            elif not self.root_agent:
                self.sep.discard(self.id)
                self.trimmed_sep.discard(self.id)
                self.sep.add(self.parent)
                self.sep.update(self.pseudo_parents)
                self.mailer.post_message(self.parent, Message(self.id, (visited, self.height + 1, self.sep), MsgType.MSG_BACKTRACK))
            else:
                DFSAgent.total_height = self.height + 1
                self.sep = set()
                self.trimmed_sep = set()
                self.pseudo_tree_created = True
                for child in self.children:
                    self.mailer.post_message(child, Message(self.id, self.level, MsgType.MSG_READY))
                self._on_pseudo_tree_created()
        elif msg.type == MsgType.MSG_READY:
            self.pseudo_tree_created = True
            self.all_parents = set(self.pseudo_parents)
            self.all_parents.add(self.parent)
            self.level = msg.content + 1
            for child in self.children:
                self.mailer.post_message(child, Message(self.id, self.level, MsgType.MSG_READY))
            self._on_pseudo_tree_created()

    def get_dot_string(self):
        dot_str = ''
        for child in self.children:
            dot_str += self.id + ' -> ' + child + ';\n'
        for child in self.pseudo_children:
            dot_str += self.id + ' -> ' + child + '  [style=dashed];\n'
        return dot_str


    @property
    def leaf_agent(self):
        return len(self.children) == 0

    @property
    def root_agent(self):
        return self.parent == -1

    def _on_pseudo_tree_created(self):
        if self.print_log:
            print(self.id, 'height', self.height, 'sep', self.sep, 'p', self.parent, 'pp', self.pseudo_parents, 'c', self.children, 'pc', self.pseudo_children)

    def _on_timestep_advanced(self):
        if self.preprocessing:
            self.preprocessing = False
            if self.id == self.mailer.root:
                child = self.degree_view.get()[-1]
                self.parent = -1
                self.children.add(child)
                visited = set()
                visited.add(self.id)
                self.mailer.post_message(child, Message(self.id, visited, MsgType.MSG_DFS))


class Message:
    def __init__(self, sender, content, msg_type=None):
        self.sender = sender
        self.content = content
        self.type = msg_type

    def __repr__(self):
        return 'sender {}, content {}'.format(self.sender, self.content)