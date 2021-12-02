from .gnn_embedding_utils import build_init_embedding, send_update, dispose_update_msg, dispose_agg_msg, dispose_val_msg
import torch.nn.functional as F


class MicroAgent:
    def __init__(self, all_layers, constraint_functions, domain, id, model, neighbor_domain, on_stop, send_message, raw_values=False):
        self.all_children = set()
        self.active = False
        self.all_layers = all_layers
        self.all_parents = set()
        self.all_rcv_embeddings = list()
        self.agg_func_embedding = None
        self.constraint_functions = constraint_functions
        self.children = set()
        self.domain = domain
        self.embedding = list()
        self.edge_index = list()
        self.function_idx = list()
        self.id = id
        self.leaf_agent = False
        self.layer_idx = 0
        self.msg_rcv_cnt = dict()
        self.model = model
        self.neighbor_domain = neighbor_domain
        self.on_stop = on_stop
        self.partial_cost = 0
        self._stop = False
        self.parent = None
        self.rcv_from = dict()
        self.ready = False
        self._send_message = send_message
        self.sep = set()
        self.sep_assign = dict()
        self.update_to = dict()
        self.val = 0
        self.raw_values = raw_values

    def stop(self):
        self._stop = True
        self.on_stop()

    def pseudo_tree_created(self):
        self.leaf_agent = len(self.all_children) == 0
        # print(f'{self.id}: {self.parent}')
        build_init_embedding(self, {})
        self.embedding = F.elu(self.all_layers[0](self.embedding.double(), self.edge_index.long()))
        self.layer_idx = 1
        self.msg_rcv_cnt['UPDATE'] = 0
        send_update(self)

    def dispose_msg(self, typ, sender, content):
        # print(f'from {sender} to {self.id}, type: {typ}')
        if typ == 'UPDATE':
            dispose_update_msg(self, sender, content, self.model)
        elif typ == 'AGG':
            dispose_agg_msg(self, content, self.model)
        elif typ == 'VAL':
            dispose_val_msg(self, content)