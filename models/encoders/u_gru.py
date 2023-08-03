import itertools
import copy
from GRU_modules import *
from torch import nn
from torch.nn import functional as F

NAN = float('nan')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LSTM(torch.nn.Module):
    def __init__(self, embedding_dim=64, hidden_dim=128, pool=None, pool_to_input=True, goal_dim=None, goal_flag=False):
        """ Initialize the LSTM forecasting model
        Attributes
        ----------
        embedding_dim : Embedding dimension of location coordinates
        hidden_dim : Dimension of hidden state of LSTM
        pool : interaction module
        pool_to_input : Bool
            if True, the interaction vector is concatenated to the input embedding of LSTM [preferred]
            if False, the interaction vector is added to the LSTM hidden-state
        goal_dim : Embedding dimension of the unit vector pointing towards the goal
        goal_flag: Bool
            if True, the embedded goal vector is concatenated to the input embedding of LSTM
        """
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.pool = pool
        self.pool_to_input = pool_to_input
        ## Location
        scale = 4.0
        self.input_embedding = InputEmbedding(16, self.embedding_dim, scale)
        """
        ## Goal
        self.goal_flag = goal_flag
        self.goal_dim = goal_dim or embedding_dim
        #self.goal_embedding = InputEmbedding(2, self.goal_dim, scale)
        goal_rep_dim = self.goal_dim if self.goal_flag else 0
        """
        ## Pooling
        pooling_dim = 0
        if pool is not None and self.pool_to_input:
            pooling_dim = self.pool.out_dim        ## LSTMs
        self.encoder1 = torch.nn.GRUCell(self.embedding_dim, self.hidden_dim)
        self.encoder2 = torch.nn.GRUCell(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.decoder = torch.nn.GRUCell(self.embedding_dim + pooling_dim, self.hidden_dim)
        self.hidden2normal = Hidden2Normal(self.hidden_dim)

    def step(self, lstm, hidden_state, obs1, obs2, batch_split, hidden_only=False,
             h_inv=None, no_pool=False, ln=None):
        """Do one step of prediction: two inputs to one normal prediction.

        Parameters
        ----------
        lstm: torch nn module [Encoder / Decoder]
            The module responsible for prediction
        hidden_state : tuple (hidden_state, cell_state)
            Current hidden_state of the pedestrians
        obs1 : Tensor [num_tracks, 2]
            Previous x-y positions of the pedestrians
        obs2 : Tensor [num_tracks, 2]
            Current x-y positions of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians

        Returns
        -------
        hidden_state : tuple (hidden_state, cell_state)
            Updated hidden_state of the pedestrians
        normals : Tensor [num_tracks, 5]
            Parameters of a multivariate normal of the predicted position
            with respect to the current position
        """

        num_tracks = len(obs2)
        # mask for pedestrians absent from scene (partial trajectories)
        # consider only the hidden states of pedestrains present in scene
        track_mask = (torch.isnan(obs1[:, 0]) + torch.isnan(obs2[:, 0])) == 0
        # print("N", num_tracks)
        # print("S", torch.sum(track_mask).item())
        # for (start, end) in zip(batch_split[:-1], batch_split[1:]):
        #    print(track_mask[start:end])

        ## Masked Hidden Cell State
        hidden_stacked = torch.stack(list(itertools.compress(hidden_state, track_mask)), dim=0)

        ## Mask current velocity & embed
        curr_velocity = obs2 - obs1
        curr_velocity = curr_velocity[track_mask]
        input_emb = self.input_embedding(curr_velocity)

        ## Mask & Pool per scene
        if self.pool is not None and not no_pool:
            hidden_states_to_pool = torch.stack(hidden_state).clone()  # detach?
            batch_pool = []
            ## Iterate over scenes
            for (start, end) in zip(batch_split[:-1], batch_split[1:]):
                ## Mask for the scene
                scene_track_mask = track_mask[start:end]
                ## Get observations and hidden-state for the scene
                prev_position = obs1[start:end][scene_track_mask]
                curr_position = obs2[start:end][scene_track_mask]
                curr_hidden_state = hidden_states_to_pool[start:end][scene_track_mask]

                ## Provide track_mask to the interaction encoders
                ## Everyone absent by default. Only those visible in current scene are present
                interaction_track_mask = torch.zeros(num_tracks, device=obs1.device).bool()
                interaction_track_mask[start:end] = track_mask[start:end]
                self.pool.track_mask = interaction_track_mask

                ## Pool
                pool_sample = self.pool(curr_hidden_state, prev_position, curr_position)
                batch_pool.append(pool_sample)

            pooled = torch.cat(batch_pool)
            if self.pool_to_input:
                input_emb = torch.cat([input_emb, pooled], dim=1)
            else:
                hidden_stacked += pooled

        if h_inv is not None:
            h_inv = torch.stack(list(itertools.compress(h_inv, track_mask)), dim=0)
            input_emb = torch.cat([input_emb, h_inv], dim=1)

        # LSTM step
        hidden_stacked = lstm(input_emb, hidden_stacked)

        if ln is not None:
            hidden_stacked = ln(hidden_stacked)

        mask_index = list(itertools.compress(range(len(track_mask)), track_mask))

        if not hidden_only:
            normal_masked = self.hidden2normal(hidden_stacked)
            # unmask [Update hidden-states and next velocities of pedestrians]
            normal = torch.full((track_mask.size(0), 5), NAN, device=obs1.device)
            for i, h, n in zip(mask_index,
                               hidden_stacked,
                               normal_masked):
                hidden_state[i] = h
                normal[i] = n
            return hidden_state, normal

        else:
            for i, h in zip(mask_index, hidden_stacked):
                hidden_state[i] = h
            return hidden_state

    def forward(self, observed, batch_split=None, prediction_truth=None, n_predict=15):
        """Forecast the entire sequence

        Parameters
        ----------
        observed : Tensor [obs_length, num_tracks, 2]
            Observed sequences of x-y coordinates of the pedestrians
        goals : Tensor [num_tracks, 2]
            Goal coordinates of the pedestrians
        batch_split : Tensor [batch_size + 1]
            Tensor defining the split of the batch.
            Required to identify the tracks of to the same scene
        prediction_truth : Tensor [pred_length - 1, num_tracks, 2]
            Prediction sequences of x-y coordinates of the pedestrians
            Helps in teacher forcing wrt neighbours positions during training
        n_predict: Int
            Length of sequence to be predicted during test time

        Returns
        -------
        rel_pred_scene : Tensor [pred_length, num_tracks, 5]
            Predicted velocities of pedestrians as multivariate normal
            i.e. positions relative to previous positions
        pred_scene : Tensor [pred_length, num_tracks, 2]
            Predicted positions of pedestrians i.e. absolute positions
        """
        # print(observed.shape)
        # print(None if prediction_truth is None else prediction_truth.shape)
        assert ((prediction_truth is None) + (n_predict is None)) == 1
        if n_predict is not None:
            # -1 because one prediction is done by the encoder already
            prediction_truth = [None for _ in range(n_predict - 1)]  # !!!

        # initialize: Because of tracks with different lengths and the masked
        # update, the hidden state for every LSTM needs to be a separate object
        # in the backprop graph. Therefore: list of hidden states instead of
        # a single higher rank Tensor.
        num_tracks = observed.size(0)
        hidden_state = [torch.zeros(self.hidden_dim, device=observed.device)
                        for _ in range(num_tracks)]

        ## Reset LSTMs of Interaction Encoders.
        if self.pool is not None:
            self.pool.reset(num_tracks, device=observed.device)

        # list of predictions
        normals = []  # predicted normal parameters for both phases
        positions = []  # true (during obs phase) and predicted positions

        if len(observed) == 2:
            positions = [observed[-1]]

        # encoder
        h_invs = []
        inv_observed = torch.flip(observed, (0,))
        for obs1, obs2 in zip(inv_observed[1:], inv_observed[:-1]):
            ##LSTM Step
            hidden_state = self.step(self.encoder1, hidden_state, obs1, obs2, batch_split,
                                     hidden_only=True, no_pool=True)  # , ln=self.ln_e1)
            h_invs.append(hidden_state)

        hidden_state = [torch.zeros(self.hidden_dim, device=observed.device)
                        for _ in range(num_tracks)]

        for obs1, obs2, h_inv in zip(observed[:-1], observed[1:], h_invs[::-1]):
            ##LSTM Step
            hidden_state = self.step(self.encoder2, hidden_state, obs1, obs2, batch_split,
                                     hidden_only=True, h_inv=h_inv, no_pool=True)  # , ln=self.ln_e2)
            # concat predictions
            # !!!normals.append(normal)
            # !!!positions.append(obs2 + normal[:, :2])  # no sampling, just mean
        # print(obs1.shape)
        # initialize predictions with last position to form velocity. DEEP COPY !!!
        prediction_truth = copy.deepcopy(list(itertools.chain.from_iterable(
            (observed[-2:], prediction_truth)
        )))

        # decoder, predictions
        i = 0
        for obs1, obs2 in zip(prediction_truth[:-1], prediction_truth[1:]):
            if obs1 is None:
                obs1 = positions[-2].detach()  # DETACH!!!
            elif i >= 2:
                for primary_id in batch_split[:-1]:
                    obs1[primary_id] = positions[-2][primary_id].detach()  # DETACH!!!
            if obs2 is None:
                obs2 = positions[-1].detach()
            elif i >= 1:
                for primary_id in batch_split[:-1]:
                    obs2[primary_id] = positions[-1][primary_id].detach()  # DETACH!!!
            hidden_state, normal = self.step(self.decoder, hidden_state, obs1, obs2, batch_split)  # , ln=self.ln_d)
            i += 1
            # concat predictions
            normals.append(normal)
            mix = nn.Linear(normal.shape[1], obs2.shape[1])
            normal = F.relu((mix(normal)))
            positions.append(obs2 + normal)  # no sampling, just mean
        # print(obs1.shape)
        # Pred_scene: Tensor [seq_length, num_tracks, 2]
        #    Absolute positions of all pedestrians
        # Rel_pred_scene: Tensor [seq_length, num_tracks, 5]
        #    Velocities of all pedestrians
        rel_pred_scene = torch.stack(normals, dim=0)
        pred_scene = torch.stack(positions, dim=0)
        # print(rel_pred_scene.shape)
        return rel_pred_scene, pred_scene


import torch

# 假设有一个包含 10 个轨迹的观测序列，每个轨迹包含 5 个时间步，每个时间步有 2 个坐标（x, y）
observed = torch.randn(64, 5, 16)

# 假设有 10 个目标行人的目标坐标

# 定义 LSTM 模型
model = LSTM(embedding_dim=16, hidden_dim=32)
    # 假设预测未来 5 个时间步的轨迹
n_predict = 5
    # 假设每个时间步预测一个轨迹模式
_, predictions = model(observed)

# 预测结果是一个字典，每个键对应一个模式的预测结果
# 在这里，假设只有一个模式，所以只有一个键
# 每个键对应一个列表，其中第一个元素是主要目标行人的预测结果，第二个元素是其他行人的预测结果
# 因为在这个例子中没有使用交互模块（pool），所以第二个元素是空的
main_target_prediction = predictions
print("Main target prediction (shape):", main_target_prediction.shape)
