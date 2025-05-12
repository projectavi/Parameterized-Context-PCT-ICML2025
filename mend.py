# ADAPTED FROM https://github.com/eric-mitchell/mend/blob/main/algs/mend.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers
import higher
import logging
from higher.patch import monkeypatch as make_functional
from collections import defaultdict
from copy import deepcopy

# from editable_model import EditableModel
# from hooks import hook_model
import aux_model_components as local_nn
# from utils import _logits, _inner_params

LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class AuxNetwork(nn.Module):
    def __init__(self, ModelClass: str, mlp_type: str, rep_dim: int, input_matrix_dim: int, rank: int, output_matrix_dim:int, internal_dim:int, combine, n_hidden, init, act, MENDrank, norm, norm_init, n_modes = None, training=True, device="cuda:0"):
        super().__init__()

        self.training = training
        self.in_dim = rep_dim
        self.input_matrix_dim = input_matrix_dim
        self.rank = rank
        self.internal_dim = internal_dim
        self.output_matrix_dim = output_matrix_dim
        self.combine = combine
        self.n_hidden = n_hidden
        self.init = init
        self.act = act
        self.MENDrank = MENDrank
        self.norm = norm
        self.norm_init = norm_init
        # self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        # self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        # self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        # self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        # self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        # self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))

        self.register_buffer("rep_mean", torch.full((rep_dim,), float("nan")))
        self.register_buffer("rep_std", torch.full((rep_dim,), float("nan")))
        self.register_buffer("rep_s", torch.full((rep_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        # MlpClass = getattr(local_nn, cfg.mlp_class)
        self.ModelClass = ModelClass
        ModelClass = getattr(local_nn, ModelClass)
        LOG.info(f"Building Auxiliary Network with MLP class {ModelClass}")
        print(f"Building Auxiliary Network with MLP class {ModelClass}")

        if self.ModelClass == "IDMLP":
            def mlp_net(in_dim, out_dim, hidden_dim):
                return ModelClass(in_dim, out_dim, hidden_dim, n_hidden, init=init, act=act, rank=MENDrank, n_modes=n_modes)

            def combined_net(activation=act):
                # print(f"Producing combined MLP with input {rep_dim}, and output {(input_matrix_dim * rank) + (output_matrix_dim * rank)}")
                return ModelClass(rep_dim, (input_matrix_dim * rank) + (output_matrix_dim * rank), rep_dim * 2,
                                n_hidden, init=init, act=activation, rank=MENDrank, n_modes=n_modes)
        elif self.ModelClass == "MLP":
            def mlp_net(in_dim, out_dim, hidden_dim):
                return ModelClass(in_dim, out_dim, hidden_dim, n_hidden, init=init, act=act, rank=MENDrank)

            def combined_net(activation=act):
                # print(f"Producing combined MLP with input {rep_dim}, and output {(input_matrix_dim * rank) + (output_matrix_dim * rank)}")
                return ModelClass(rep_dim, (input_matrix_dim * rank) + (output_matrix_dim * rank), rep_dim * 2,
                                n_hidden, init=init, act=activation, rank=MENDrank)

        elif self.ModelClass == "RNN":
            def mlp_net(in_dim, out_dim, hidden_dim, activation):
                return ModelClass(in_dim, out_dim, hidden_dim, n_hidden, init=init, act=activation, rank=MENDrank, mlp_type=mlp_type)

            def combined_net(activation=act):
                # print(f"Producing combined MLP with input {rep_dim}, and output {(input_matrix_dim * rank) + (output_matrix_dim * rank)}")
                return ModelClass(rep_dim, (input_matrix_dim * rank) + (output_matrix_dim * rank), rep_dim * 2,
                                n_hidden, init=init, act=activation, rank=MENDrank, mlp_type=mlp_type)
            
        elif self.ModelClass == "GRU":
            def mlp_net(in_dim, out_dim, hidden_dim, activation):
                return ModelClass(in_dim, out_dim, hidden_dim, n_hidden, init=init, act=activation, rank=MENDrank, mlp_type=mlp_type)

            def combined_net(activation=act):
                # print(f"Producing combined MLP with input {rep_dim}, and output {(input_matrix_dim * rank) + (output_matrix_dim * rank)}")
                return ModelClass(rep_dim, (input_matrix_dim * rank) + (output_matrix_dim * rank), self.internal_dim,
                                n_hidden, init=init, act=activation, rank=MENDrank, mlp_type=mlp_type)
            
        if combine:
            self.mlp = combined_net(activation=act)
        else:
            self.mlpinput_matrix_, self.mlpoutput_matrix_ = mlp_net(rep_dim, (input_matrix_dim * rank), self.internal_dim, activation=act), mlp_net(rep_dim, (output_matrix_dim * rank), self.internal_dim, activation=act)

        self.old_parameters = deepcopy(self.parameters())
        self.old_parameters = [p.clone().detach().to(device) for p in self.old_parameters]

    def forward(self, rep, param_idx=None):
        rep = rep.to(torch.float32)

        # print("FORWARD")
        # print(rep.shape)

        # print(rep.isnan())

        rep_ = rep.view(-1, rep.shape[-1])

        # print(rep.shape)

        # nz_mask = (rep_ != 0).any(-1)
        # rep_ = rep_[nz_mask]

        # # print(rep_.isnan())

        # # print(rep_.shape)

        # # print("AUX: forward triggered")
        # # print(f"self.training {self.training}")

        # # FLAG - this was only enabled when self.training was true, possible that mean should stay constant when training is not ocurring
        # # TODO - figure out this bug, minor though - more effort later
        # # if self.training:
        # # print("AUX: If training triggered")
        # if self.ModelClass != "RNN" and self.ModelClass != "GRU":
        #     for idx in range(rep_.shape[0]):
        #         # print("AUX: Rep iter triggered")
        #         if not self.norm_init:
        #             self.rep_mean = rep_[idx].clone().detach()
        #             self.rep_s.zero_()
        #             self.k[:] = 1
        #             self.norm_init = True
        #         else:
        #             self.k += 1
        #             self.rep_mean, self.rep_s = update_counter(rep_[idx], self.rep_mean, self.rep_s, self.k)

        #     if self.k < 2:
        #         # raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
        #         print(f"Warning: Can't perform normalization with only {self.k} samples so far, disabling norm")
        #         rep_input = rep_
        #         self.norm = False
        #     else:
        #         self.rep_std = (self.rep_s / (self.k - 1)) ** 0.5
        #         if self.norm:
        #             rep_input = (rep_ - self.rep_mean) / (self.rep_std + 1e-7)
        #         else:
        #             rep_input = rep_
        # else:
        rep_input = rep_
            
        # print(rep_input.shape)
        # self.old_parameters = deepcopy(self.parameters())

         # Assert that all parameters are still trainable
         # Just to check that training is still running
        if self.combine:
            assert(all(p.requires_grad for p in self.mlp.parameters()))
        else:
            assert(all(p.requires_grad for p in self.mlpinput_matrix_.parameters()) and all(p.requires_grad for p in self.mlpoutput_matrix_.parameters()))

        if self.combine:
            if self.ModelClass == "IDMLP":
                output = self.mlp(rep_input, mode=param_idx)
            elif self.ModelClass == "RNN":
                output = self.mlp(rep_input, init="xavier_uniform")
            else:
                output = self.mlp(rep_input)
            # print(f"Actual MLP output shape {output.shape}")
            lora_input_matrix_, lora_output_matrix_ = output.split([(self.input_matrix_dim * self.rank), (self.output_matrix_dim * self.rank)], -1)
            return lora_input_matrix_, lora_output_matrix_
        else:
            if self.ModelClass == "IDMLP":
                return self.mlpinput_matrix_(rep_input, mode=param_idx), self.mlpoutput_matrix_(rep_input, mode=param_idx)
            elif self.ModelClass == "RNN":
                return self.mlpinput_matrix_(rep_input, init="xavier_uniform"), self.mlpoutput_matrix_(rep_input, init="xavier_uniform")
            else:
                return self.mlpinput_matrix_(rep_input), self.mlpoutput_matrix_(rep_input)
            
    def parameters(self, recurse = True):
        if self.combine:
            # print("Combining MLP")
            # print(list(self.mlp.parameters()))
            # print(list(super().parameters(recurse=recurse)))
            # exit(0)
            return list(super().parameters(recurse=recurse)) + list(self.mlp.parameters())
        else:
            # print("Not combining MLP")
            # print(list(self.mlpinput_matrix_.parameters()))
            # print(list(self.mlpoutput_matrix_.parameters()))
            # print(list(super().parameters(recurse=recurse)))
            # exit(0)
            return list(super().parameters(recurse=recurse)) + list(self.mlpinput_matrix_.parameters()) + list(self.mlpoutput_matrix_.parameters())