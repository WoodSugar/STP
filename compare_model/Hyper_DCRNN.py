# -*- coding: utf-8 -*-
""" 
@Time   : 2024/04/10
 
@Author : Shen Fang
"""

import torch
import torch.nn as nn
import numpy as np
from main.main_arch.hypergraph import HyperNet


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class LayerParams:
    """Layer parameters."""

    def __init__(self, rnn_network: torch.nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict = {}
        self._biases_dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = torch.nn.Parameter(torch.empty(*shape))
            torch.nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                '{}_weight_{}'.format(self._type, str(shape)),
                nn_param)
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = torch.nn.Parameter(torch.empty(length))
            torch.nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                '{}_biases_{}'.format(self._type, str(length)), biases)

        return self._biases_dict[length]


class HyperDCGRUCell(torch.nn.Module):
    def __init__(self, num_units, in_dim, adj_mx, max_diffusion_step, num_nodes, Sk, Tk, nonlinearity='tanh', use_gc_for_ru=True):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        # support other nonlinearities up here?
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._in_dim = in_dim
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        # for support in supports:
        # self._supports.append(self._build_sparse_matrix(support))
        self._supports = adj_mx

        self._fc_params = LayerParams(self, 'fc')
        self._gconv_params = LayerParams(self, 'hypergconv')
        # self.ru_gcn = HyperNet(num_nodes, 1, in_dim, Sk, Tk)
        self.gcn = HyperNet(num_nodes, 1, in_dim, Sk, Tk)
        # num_nodes: int, temporal_length: int, input_dim: int

    def forward(self, inputs, hx):
        output_size = 2 * self._num_units
        if self._use_gc_for_ru:
            fn = self._hypergconv
        else:
            fn = self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            tensor=value, split_size_or_sections=self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._hypergconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights(
            (input_size, output_size)).to(inputs_and_state.device)
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value = value + biases.to(inputs_and_state.device)
        return value

    def _hypergconv(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))

        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state  # [B, N, C+D]
        x = x.unsqueeze(1) # [B, T=1, N, C+D]

        x = self.gcn(x)

        y = x.squeeze(1)  # [B, N, C+D]

        weights = self._gconv_params.get_weights(
            (input_size, output_size)).to(x.device)
        
        y = torch.einsum('bki,ij->bkj', y, weights)

        biases = self._gconv_params.get_biases(
            output_size, bias_start).to(x.device)
        
        y += biases

        return torch.reshape(y, [batch_size, self._num_nodes * output_size])


        x0 = x.permute(1, 2, 0)  # (N, C+D, B)
        x0 = torch.reshape(
            x0, shape=[self._num_nodes, input_size * batch_size])  # [N, (C+D)*B]
        x = torch.unsqueeze(x0, 0)  # [1, N, (C+D)*B]

        if self._max_diffusion_step == 0:
            pass
        else:
            pass
        x = x # [Step, N, (C+D)*B]


    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(
            x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.mm(support.to(x0.device), x0)
                x = self._concat(x, x1)

                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.mm(support.to(x0.device), x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        # Adds for x itself.
        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size)).to(x.device)
        # (batch_size * self._num_nodes, output_size)
        x = torch.matmul(x, weights)

        biases = self._gconv_params.get_biases(
            output_size, bias_start).to(x.device)
        x += biases
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])



class Seq2SeqAttrs:
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx
        self.max_diffusion_step = int(
            model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units
        self.Sk = int(model_kwargs.get("Sk", 20))
        self.Tk = int(model_kwargs.get("Tk", 1))
        self.use_gc_for_ru = model_kwargs.get("use_gc_for_ru")

class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))  # for the encoder
        

        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(HyperDCGRUCell(self.rnn_units, 1+ self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk, use_gc_for_ru=self.use_gc_for_ru))

        for _ in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(HyperDCGRUCell(self.rnn_units, 2*self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk, use_gc_for_ru=self.use_gc_for_ru))
        
        # self.dcgru_layers = nn.ModuleList(
        #     [HyperDCGRUCell(self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size)).to(inputs.device)
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        # runs in O(num_layers) so not too slow
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, adj_mx, **model_kwargs):
        # super().__init__(is_training, adj_mx, **model_kwargs)
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))  # for the decoder
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)

        self.dcgru_layers = nn.ModuleList()
        self.dcgru_layers.append(HyperDCGRUCell(self.rnn_units, 1 + self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk, use_gc_for_ru=self.use_gc_for_ru))

        for _ in range(1, self.num_rnn_layers):
            self.dcgru_layers.append(HyperDCGRUCell(self.rnn_units, 2*self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk, use_gc_for_ru=self.use_gc_for_ru))
        
        # self.dcgru_layers = nn.ModuleList(
        #     [HyperDCGRUCell(self.rnn_units, self.rnn_units, adj_mx, self.max_diffusion_step, self.num_nodes, self.Sk, self.Tk) for _ in range(self.num_rnn_layers)])

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state

        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)

        return output, torch.stack(hidden_states)

class DCRNN_HyperGConv(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting
    Link: https://arxiv.org/abs/1707.01926
    Codes are modified from the official repo:
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_cell.py,
        https://github.com/chnsh/DCRNN_PyTorch/blob/pytorch_scratch/model/pytorch/dcrnn_model.py
    """

    def __init__(self, adj_mx, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, adj_mx, **model_kwargs)
        self.encoder_model = EncoderModel(adj_mx, **model_kwargs)
        self.decoder_model = DecoderModel(adj_mx, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False))

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state)

        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim)).to(encoder_hidden_state.device)
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol

        outputs = []

        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state)
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, **kwargs) -> torch.Tensor:
        """Feedforward function of DCRNN.

        Args:
            history_data (torch.Tensor): history data with shape [L, B, N*C]
            future_data (torch.Tensor, optional): future data with shape [L, B, N*C_out]
            batch_seen (int, optional): batches seen till now, used for curriculum learning. Defaults to None.

        Returns:
            torch.Tensor: prediction with shape [L, B, N*C_out]
        """

        # reshape data
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
        history_data = history_data.transpose(0, 1)         # [L, B, N*C]

        if future_data is not None:
            # future_data = future_data[..., [0]]     # teacher forcing only use the first dimension.
            batch_size, length, num_nodes, channels = future_data.shape
            future_data = future_data.reshape(batch_size, length, num_nodes * channels)      # [B, L, N*C]
            future_data = future_data.transpose(0, 1)         # [L, B, N*C]

        # DCRNN
        encoder_hidden_state = self.encoder(history_data)
        outputs = self.decoder(encoder_hidden_state, future_data,
                               batches_seen=batch_seen)      # [L, B, N*C_out]

        # reshape to B, L, N, C
        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(B, L, self.num_nodes,
                               self.decoder_model.output_dim)

        if batch_seen == 0:
            print("Warning: decoder only takes the first dimension as groundtruth.")
            print("Parameter Number: ".format(count_parameters(self)))
            print(count_parameters(self))
        return outputs


class HyperDCRNN(DCRNN_HyperGConv):
    def __init__(self, adj_mx, **model_kwargs):
        super().__init__(adj_mx, **model_kwargs)
        self.use_long_history = model_kwargs.get("use_long_history", False)

    def forward(self, history_data: torch.Tensor, long_history_data: torch.Tensor, future_data: torch.Tensor, epoch: int, batch_seen: int, train: bool, **kwargs):
        """ Feedforward function for DCRNN.
            history_data (torch.Tensor): inputs with shape [B, L1, N, C].
            long_history_data (torch.Tensor): inputs with shape [B, L1 * P, N, C].
            future_data (torch.Tensor) : inputs with shape [B, L2, N, C].
        """
        # B, src_len, N, C = history_data.size()
        # history_data = history_data.transpose(1, 2)  # [L, B, N, C]
        if self.use_long_history:
            return super().forward(long_history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)
        else:
            return super().forward(history_data, future_data, batch_seen=batch_seen, train=train, **kwargs)