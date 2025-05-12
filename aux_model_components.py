# ADAPTED from https://github.com/eric-mitchell/mend/blob/main/nn.py

import torch
import torch.nn as nn

import logging

LOG = logging.getLogger(__name__)

class GRU(nn.Module):
    def __init__(
            self,
            indim: int,
            outdim: int,
            hidden_dim: int,
            n_hidden: int,
            init: str = None,
            act: str = None,
            rank: int = None,
            n_modes: int = None,
            mlp_type: str = None,
            device: str = "cuda:0",
    ):
        super(GRU, self).__init__()
        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim
        self.init = init
        self.device = device
        self.gru = nn.GRU(
            input_size=indim, 
            hidden_size=hidden_dim, 
            num_layers=n_hidden,
        ).train()
        self.mlp_type = mlp_type
        if self.mlp_type == "MLP":
            self.fc = MLP(hidden_dim, outdim, hidden_dim, n_hidden, init, act, rank)
        elif self.mlp_type == "IDMLP":
            self.fc = IDMLP(hidden_dim, outdim, hidden_dim, n_hidden, init, act, rank)
    
    def forward(self, x):
        hidden_state = self.init_hidden(self.init)
        output, hidden_state = self.gru(x, hidden_state)
        output = self.fc(output[-1])
        return output
    
    def init_hidden(self, init):
        if init == "kaiming_uniform":
            tensor = nn.init.kaiming_uniform_(torch.empty(1, self.hidden_dim))
        elif init == "kaiming_normal":
            tensor = nn.init.kaiming_normal_(torch.empty(1, self.hidden_dim))
        elif init == "xavier_uniform":
            tensor = nn.init.xavier_uniform_(torch.empty(1, self.hidden_dim))
        elif init == "xavier_normal":
            tensor = nn.init.xavier_normal_(torch.empty(1, self.hidden_dim))

        return tensor.to(self.device)

class RNN(nn.Module):
    def __init__(
            self,
            indim: int,
            outdim: int,
            hidden_dim: int,
            n_hidden: int,
            init: str = None,
            act: str = None,
            rank: int = None,
            n_modes: int = None,
            mlp_type: str = None,
            device: str = "cuda:0",
    ):
        
        super(RNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.init = init
        self.device = device

        if mlp_type == "MLP":
            self.lin1 = MLP(indim + hidden_dim, hidden_dim, hidden_dim, n_hidden, init, act, rank)
            self.lin2 = MLP(indim + hidden_dim, outdim, hidden_dim, n_hidden, init, act, rank)
        elif mlp_type == "IDMLP":
            self.lin1 = IDMLP(indim + hidden_dim, hidden_dim, hidden_dim, n_hidden, init, act, rank, n_modes)
            self.lin2 = IDMLP(indim + hidden_dim, outdim, hidden_dim, n_hidden, init, act, rank, n_modes)

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "tanh":
            self.act = nn.Tanh()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU()

    def parameters(self, recurse = True):
        # print("get params called")
        return list(self.lin1.parameters()) + list(self.lin2.parameters()) + list(super().parameters(recurse=recurse))

    def forward(self, input, init):
        
        if len(input.shape) == 3:
            _, seq_len, _ = input.shape
        elif len(input.shape) == 2:
            seq_len, _ = input.shape
        else:
            raise ValueError(f"Expected Input as 3D Tensor, Got {len(input.shape)} dimensions.")
        
        hidden_state = self.init_hidden(init)
        # print(hidden_state.isnan())

        # print(input.isnan())
        # exit(0)

        if len(input.shape) == 3:
            for i in range(seq_len):
                x = input[:, i, :].unsqueeze(0)
                # print("ITER")
                # print(x.shape)
                # print(hidden_state.shape)
                combined = torch.cat((x, hidden_state), -1)
                hidden_state = self.act(self.lin1(combined))
                output = self.lin2(combined)
        elif len(input.shape) == 2:
            for i in range(seq_len):
                x = input[i, :].unsqueeze(0)
                # print("ITER")
                # print(x.shape)
                # print(hidden_state.shape)
                # print(x.isnan())
                combined = torch.cat((x, hidden_state), -1)
                # print(combined.shape)
                hidden_state = self.act(self.lin1(combined))
                output = self.lin2(combined)
                # print(hidden_state.isnan())
                # print(output.isnan())
                # exit(0)

        return output
    
    def init_hidden(self, init):
        if init == "kaiming_uniform":
            tensor = nn.init.kaiming_uniform_(torch.empty(1, self.hidden_dim))
        elif init == "kaiming_normal":
            tensor = nn.init.kaiming_normal_(torch.empty(1, self.hidden_dim))
        elif init == "xavier_uniform":
            tensor = nn.init.xavier_uniform_(torch.empty(1, self.hidden_dim))
        elif init == "xavier_normal":
            tensor = nn.init.xavier_normal_(torch.empty(1, self.hidden_dim))

        return tensor.to(self.device)


class IDMLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = None,
        act: str = None,
        rank: int = None,
        n_modes: int = None,
        device: str = "cuda:0",
    ):
        self.hidden_dim = hidden_dim
        super().__init__()
        LOG.info(f"Building IDMLP ({init}) {[indim] * (n_hidden + 2)}")
        self.layers = [LRLinear(indim, hidden_dim, rank=rank, relu=True, init=init, n_modes=n_modes)]
        for idx in range(n_hidden + 1):
            self.layers.append(
                LRLinear(hidden_dim, hidden_dim, rank=rank, relu=idx < n_hidden, init=init, n_modes=n_modes),
            )
        self.layers.append(LRLinear(hidden_dim, outdim, rank=rank, relu=False, init="xavier", n_modes=n_modes))

        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, mode=None):
        old = None
        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            if idx == 0:
                x = layer(x, mode=mode)
            elif idx % 2 == 1:
                old = x
                x = layer(x, mode=mode)
            else:
                x = layer(x + old, mode=mode)

        return x


class LRLinear(nn.Module):
    def __init__(self, inf, outf, rank: int = None, relu=False, init="id", n_modes=None):
        super().__init__()
        self.indim = inf
        mid_dim = min(rank, inf)
        if init == "id":
            self.u = nn.Parameter(torch.zeros(outf, mid_dim))
            self.v = nn.Parameter(torch.randn(mid_dim, inf))
        elif init == "xavier" or init == "xavier_uniform":
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.xavier_uniform_(self.u.data, gain=nn.init.calculate_gain("relu"))
            nn.init.xavier_uniform_(self.v.data, gain=1.0)
        elif init == "kaiming_uniform":
            self.u = nn.Parameter(torch.empty(outf, mid_dim))
            self.v = nn.Parameter(torch.empty(mid_dim, inf))
            nn.init.kaiming_uniform_(self.u.data)
            nn.init.kaiming_uniform_(self.v.data)
        else:
            raise ValueError(f"Unrecognized initialization {init}")

        if n_modes is not None:
            self.mode_shift = nn.Embedding(n_modes, outf)
            self.mode_shift.weight.data.zero_()
            self.mode_scale = nn.Embedding(n_modes, outf)
            self.mode_scale.weight.data.fill_(1)

        self.n_modes = n_modes
        self.bias = nn.Parameter(torch.zeros(outf))
        self.inf = inf
        self.init = init

    def forward(self, x, mode=None):
        if mode is not None:
            assert self.n_modes is not None, "Linear got a mode but wasn't initialized for it"
            assert mode < self.n_modes, f"Input mode {mode} outside of range {self.n_modes}"
        assert x.shape[-1] == self.inf, f"Input wrong dim ({x.shape}, {self.inf})"

        pre_act = (self.u @ (self.v @ x.T)).T
        if self.bias is not None:
            pre_act += self.bias

        if mode is not None:
            if not isinstance(mode, torch.Tensor):
                mode = torch.tensor(mode).to(x.device)
            scale, shift = self.mode_scale(mode), self.mode_shift(mode)
            pre_act = pre_act * scale + shift

        # need clamp instead of relu so gradient at 0 isn't 0
        acts = pre_act.clamp(min=0)
        if self.init == "id":
            return acts + x
        else:
            return acts


class MLP(nn.Module):
    def __init__(
        self,
        indim: int,
        outdim: int,
        hidden_dim: int,
        n_hidden: int,
        init: str = "xavier_uniform",
        act: str = "relu",
        rank: int = None,
        device: str = "cuda:0",
    ):
        super().__init__()

        self.init = init

        if act == "relu":
            self.act = nn.ReLU()
        elif act == "learned":
            self.act = ActMLP(10, 1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        elif act == "gelu":
            self.act = nn.GELU()
        elif act == "tanh":
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unrecognized activation function '{act}'")

        if hidden_dim is None:
            hidden_dim = outdim * 2

        if init.startswith("id") and outdim != indim:
            LOG.info(f"Overwriting outdim ({outdim}) to be indim ({indim})")
            outdim = indim

        if init == "id":
            old_hidden_dim = hidden_dim
            if hidden_dim < indim * 2:
                hidden_dim = indim * 2

            if hidden_dim % indim != 0:
                hidden_dim += hidden_dim % indim

            if old_hidden_dim != hidden_dim:
                LOG.info(
                    f"Overwriting hidden dim ({old_hidden_dim}) to be {hidden_dim}"
                )

        if init == "id_alpha":
            self.alpha = nn.Parameter(torch.zeros(1, outdim))

        dims = [indim] + [hidden_dim] * n_hidden + [outdim]
        LOG.info(f"Building ({init}) MLP: {dims} (rank {rank})")

        layers = []
        for idx, (ind, outd) in enumerate(zip(dims[:-1], dims[1:])):
            if rank is None:
                layers.append(nn.Linear(ind, outd))
            else:
                layers.append(LRLinear(ind, outd, rank=rank, init="xavier"))
            if idx < n_hidden:
                layers.append(self.act)

        if rank is None:
            if init == "id":
                if n_hidden > 0:
                    layers[0].weight.data = torch.eye(indim).repeat(
                        hidden_dim // indim, 1
                    )
                    layers[0].weight.data[hidden_dim // 2:] *= -1
                    layers[-1].weight.data = torch.eye(outdim).repeat(
                        1, hidden_dim // outdim
                    )
                    layers[-1].weight.data[:, hidden_dim // 2:] *= -1
                    layers[-1].weight.data /= (hidden_dim // indim) / 2.0

            for layer in layers:
                if isinstance(layer, nn.Linear):
                    if init == "ortho":
                        nn.init.orthogonal_(layer.weight)
                    elif init == "id":
                        if layer.weight.shape[0] == layer.weight.shape[1]:
                            layer.weight.data = torch.eye(hidden_dim)
                    else:
                        gain = 3 ** 0.5 if (layer is layers[-1]) else 1.0
                        nn.init.xavier_uniform_(layer.weight, gain=gain)

                    layer.bias.data[:] = 0

        layers[-1].bias = None
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.init == "id_alpha":
            return x + self.alpha * self.mlp(x)
        else:
            return self.mlp(x)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    m0 = MLP(1000, 1000, 1500, 3)
    m1 = MLP(1000, 1000, 1500, 3, init="id")
    m2 = MLP(1000, 1000, 1500, 3, init="id_alpha")
    m3 = MLP(1000, 1000, 1500, 3, init="ortho", act="learned")

    x = 0.01 * torch.randn(999, 1000)

    y0 = m0(x)
    y1 = m1(x)
    y2 = m2(x)
    y3 = m3(x)

    print("y0", (y0 - x).abs().max())
    print("y1", (y1 - x).abs().max())
    print("y2", (y2 - x).abs().max())
    print("y3", (y3 - x).abs().max())

    assert not torch.allclose(y0, x)
    assert torch.allclose(y1, x)
    assert torch.allclose(y2, x)
    assert not torch.allclose(y3, x)
    import pdb; pdb.set_trace()  # fmt: skip