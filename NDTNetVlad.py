from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .NetVLAD import NetVLADLoupe
from .Transformer_skip import Transformer


class NDTNetVlad(nn.Module):
    def __init__(self, num_points=2000, output_dim=256, emb_dims=1024, layer_number=3):
        super(NDTNetVlad, self).__init__()
        self.Transformer = Transformer(num_points=num_points, emb_dims=emb_dims, layer_number=layer_number)
        self.net_vlad = NetVLADLoupe(feature_size=1024, max_samples=num_points, cluster_size=64,
                                     output_dim=output_dim, gating=True, add_batch_norm=True,
                                     is_training=True)

    def forward(self, x):
        x = self.Transformer(x)
        x = self.net_vlad(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_points = 2000
    batch_size = 15
    sim_data = Variable(torch.rand(batch_size, 1, num_points, 3))
    sim_data = sim_data.cuda()

    nnv = NDTNetVlad(num_points=num_points, output_dim=256, emb_dims=1024, layer_number=3).cuda()
    nnv.train()
    out3 = nnv(sim_data)
    print('pnv', out3.size())
