import torch


class Bucket:
    def __init__(self, data, dims):
        self.data = data
        self.dims = dims

    def align_(self, target_dims):
        assert len(self.data.size()) <= len(target_dims)
        dims = list(self.dims)
        while len(self.data.size()) < len(target_dims):
            self.data.unsqueeze_(-1)
        for d in target_dims:
            if d not in dims:
                dims.append(d)
        relocate = []
        for d in target_dims:
            idx = dims.index(d)
            relocate.append(idx)
        self.data = self.data.permute(relocate)
        self.dims = target_dims

    def squeeze(self):
        dims = []
        for i in range(len(self.dims)):
            if self.data.shape[i] != 1:
                dims.append(self.dims[i])
        self.data = self.data.squeeze()
        self.dims = dims

    def reduce(self, assign, eval=False):
        remaining_dim = []
        assign_tuple = []
        assign_dim = []
        for dim in self.dims:
            if dim in assign:
                assign_dim.append(dim)
                assign_tuple.append(assign[dim])
            else:
                remaining_dim.append(dim)
        self.align_(assign_dim + remaining_dim)
        if not eval:
            data = self.data[tuple(assign_tuple)]
            return Bucket(data, remaining_dim)
        else:
            assert len(remaining_dim) == 0
            return self.data[tuple(assign_tuple)].item()

    def proj(self, target):
        assert target in self.dims
        idx = self.dims.index(target)
        dims = list(self.dims)
        data, _ = self.data.min(idx)
        dims.pop(idx)
        return Bucket(data, dims)

    @classmethod
    def from_matrix(cls, matrix, row, col):
        data = torch.tensor(matrix)
        dims = [row, col]
        return Bucket(data, dims)

    @classmethod
    def join(cls, buckets):
        target_dims = set()
        for bucket in buckets:
            target_dims.update(bucket.dims)
        target_dims = list(target_dims)
        for bucket in buckets:
            bucket.align_(target_dims)
        data = buckets[0].data
        for i in range(1, len(buckets)):
            data = data + buckets[i].data
        return Bucket(data, target_dims)