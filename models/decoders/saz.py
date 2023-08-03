from models.decoders.decoder import PredictionDecoder
import torch
import torch.nn as nn
from typing import Dict, Union
from models.decoders.utils import cluster_traj
import sys
sys.path.append('/home/xie/code/PGP1/models/decoders')
import enc_z_att
# Initialize device:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SAZ(PredictionDecoder):

    def __init__(self, args):
        """
        Latent variable conditioned decoder.

        args to include:
        agg_type: 'combined' or 'sample_specific'. Whether we have a single aggregated context vector or sample-specific
        num_samples: int Number of trajectories to sample
        op_len: int Length of predicted trajectories
        lv_dim: int Dimension of latent variable
        encoding_size: int Dimension of encoded scene + agent context
        hidden_size: int Size of output mlp hidden layer
        num_clusters: int Number of final clustered trajectories to output

        """
        super().__init__()
        self.agg_type = args['agg_type']
        self.num_samples = args['num_samples']
        self.op_len = args['op_len']
        self.lv_dim = args['lv_dim']
        self.hidden = nn.Linear(args['encoding_size'] + args['lv_dim'], args['hidden_size'])
        self.op_traj = nn.Linear(args['hidden_size'], args['op_len'] * 2)
        self.leaky_relu = nn.LeakyReLU()
        self.num_clusters = args['num_clusters']

    def forward(self, inputs: Union[Dict, torch.Tensor], encodings_) -> Dict:
        """
        Forward pass for latent variable model.

        :param inputs: aggregated context encoding,
         shape for combined encoding: [batch_size, encoding_size]
         shape if sample specific encoding: [batch_size, num_samples, encoding_size]
        :return: predictions
        """
        node_encodings = encodings_['context_encoding']['combined']
        max_nodes = node_encodings.shape[1]
        node_encodings = node_encodings[:, 0:1, :].repeat(1, self.num_samples, 1)
        # node_encodings = torch.cat((node_encodings, node_encodings), dim=1)
        lv_dim = node_encodings.shape[2]


        if type(inputs) is torch.Tensor:
            agg_encoding = inputs
        else:
            agg_encoding = inputs['agg_encoding']

        if self.agg_type == 'combined':
            agg_encoding = agg_encoding.unsqueeze(1).repeat(1, self.num_samples, 1)
        else:
            if len(agg_encoding.shape) != 3 or agg_encoding.shape[1] != self.num_samples:
                raise Exception('Expected ' + str(self.num_samples) + 'encodings for each train/val data')

        # 自己添加
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, max_nodes + self.num_samples, lv_dim, device=device)  # 生成潜在变量[64, 164, 1000, 5]
        enc_shape_2 = node_encodings.shape[2]  # 32
        e_z_att = enc_z_att.Attention(enc_size=enc_shape_2, z_size=lv_dim, attention_size=64).to(device)
        z = e_z_att(node_encodings, z)  # 32,1000,64
        z_shape_2 = z.shape[2]  # 64
        Linear = nn.Linear(in_features=z_shape_2, out_features=lv_dim).to(device)
        z = self.leaky_relu(Linear(z))
        Linear1 = nn.Linear(max_nodes + self.num_samples, self.num_samples).to(device)
        z = Linear1(z.permute(0, 2, 1))
        z = self.leaky_relu(z.permute(0, 2, 1))
        # agg_encoding = torch.cat((agg_encoding, node_encodings1), dim=1)
        agg_encoding2 = torch.cat((agg_encoding, z), dim=-1)  # 将潜在变量和聚合编码进行融合[16, 1000, 165]
        h = self.leaky_relu(self.hidden(agg_encoding2))  # 进行压缩将维[16, 1000, 128]

        # Sample latent variable and concatenate with aggregated encoding
        batch_size = agg_encoding.shape[0]
        z = torch.randn(batch_size, self.num_samples, self.lv_dim, device=device)
        agg_encoding1 = torch.cat((agg_encoding, z), dim=2)
        h1 = self.leaky_relu(self.hidden(agg_encoding1))
        h = h1 + h
        # h = h1

        # Output trajectories
        traj = self.op_traj(h)
        traj = traj.reshape(batch_size, self.num_samples, self.op_len, 2)

        # Cluster
        traj_clustered, probs = cluster_traj(self.num_clusters, traj)

        predictions = {'traj': traj_clustered, 'probs': probs}

        if type(inputs) is dict:
            for key, val in inputs.items():
                if key != 'agg_encoding':
                    predictions[key] = val

        return predictions
