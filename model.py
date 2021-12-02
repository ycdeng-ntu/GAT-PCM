import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_scatter import scatter_sum


class GATNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(64, 8, heads=8)
        self.conv3 = GATConv(64, 8, heads=8)
        self.conv4 = GATConv(64, out_channels, heads=4, concat=False)
        self.agg = nn.Linear(out_channels, out_channels, bias=False)
        self.transform = nn.Linear(out_channels, out_channels, bias=False)
        self.out = nn.Linear(out_channels * 2, 1)

    def forward(self, batch, decision_var_idxes):
        function_idxes = []
        s = 0
        flag = []
        for i in range(batch.num_graphs):
            data = batch.get_example(i)
            function_idxes += [j + s for j in data.function_idx]
            flag += [i] * len(data.function_idx)
            decision_var_idxes[i] += s
            s += data.x.shape[0]
        flag = torch.tensor(flag, device=batch.x.device)

        x = self.conv1(batch.x, batch.edge_index)
        x = F.elu(x)
        x = self.conv2(x, batch.edge_index)
        x = F.elu(x)
        x = self.conv3(x, batch.edge_index)
        x = F.elu(x)
        x = F.elu(self.conv4(x, batch.edge_index))

        agg = x[function_idxes]
        agg = scatter_sum(agg, flag, dim=0)
        agg = self.agg(agg)
        trans = x[decision_var_idxes, :]
        trans = self.transform(trans)
        return self.out(F.elu(torch.cat([agg, trans], dim=1)))


    @torch.no_grad()
    def inference(self, x, edge_index, decision_var_idxes, function_idx, functions=None, dv=None):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = self.conv3(x, edge_index)
        x = F.elu(x)
        x = F.elu(self.conv4(x, edge_index))

        agg = x[function_idx].sum(dim=0)
        agg = self.agg(agg)

        agg = agg.repeat((len(decision_var_idxes), 1))

        trans = x[decision_var_idxes, :]
        trans = self.transform(trans)

        return self.out(F.elu(torch.cat([agg, trans], dim=1)))
