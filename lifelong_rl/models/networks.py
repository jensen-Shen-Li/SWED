import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from lifelong_rl.torch import pytorch_util as ptu
from lifelong_rl.torch.modules import LayerNorm

import copy

def identity(x):
    return x




class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        self.fcs = []
        self.batch_norms = []

        # data normalization
        self.input_mu = nn.Parameter(ptu.zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(ptu.ones(input_size), requires_grad=False).float()

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            hidden_init(fc.weight, w_scale)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.batch_norm:
                bn = nn.BatchNorm1d(next_size)
                self.__setattr__('batch_norm%d' % i, bn)
                self.batch_norms.append(bn)

            in_size = next_size

        self.last_fc = nn.Linear(in_size, output_size)
        if final_init_scale is None:
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        else:
            ptu.orthogonal_init(self.last_fc.weight, final_init_scale)
            self.last_fc.bias.data.fill_(0)

    def forward(self, input, return_preactivations=False):
        h = (input - self.input_mu) / (self.input_std + 1e-6)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
            if self.layer_norm is not None:
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std != std] = 0
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std = mask * std + (1-mask) * np.ones(self.input_size)
        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class Ensemble(nn.Module):

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList()
        for m in models:
            self.models.append(m)

        self.ensemble_size = len(models)
        self.input_size = self.models[0].input_size
        self.output_size = self.models[0].output_size

    def forward(self, input):
        preds = ptu.zeros((len(self.models), *input.shape[:-1], self.output_size))
        for i in range(len(self.models)):
            preds[i] = self.models[i].forward(input)
        return preds

    def sample(self, input):
        preds = self.forward(input)
        inds = torch.randint(0, len(self.models), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])
        samples = (inds == 0).float() * preds[0]
        for i in range(1, len(self.models)):
            samples += (inds == i).float() * preds[i]
        return samples

    def fit_input_stats(self, data, mask=None):
        for m in self.models:
            m.fit_input_stats(data, mask=mask)


class ParallelizedLayer(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsemble(nn.Module):

    def __init__(
            self,
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            b_init_value=0.0,
            layer_norm=False,
            layer_norm_kwargs=None,
            spectral_norm=False,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        # data normalization
        self.input_mu = nn.Parameter(
            ptu.zeros(input_size), requires_grad=False).float()
        self.input_std = nn.Parameter(
            ptu.ones(input_size), requires_grad=False).float()

        self.fcs = []

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            layer_size = (ensemble_size, in_size, next_size)
            fc = ParallelizedLayer(
                ensemble_size, in_size, next_size,
                w_std_value=1/(2*np.sqrt(in_size)),
                b_init_value=b_init_value,
            )
            if spectral_norm:
                fc = nn.utils.spectral_norm(fc, name='W')
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayer(
            ensemble_size, in_size, output_size,
            w_std_value=1/(2*np.sqrt(in_size)),
            b_init_value=b_init_value,
        )

    def forward(self, input):
        dim = len(input.shape)

        # input normalization
        h = (input - self.input_mu) / self.input_std

        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            h = h.unsqueeze(0)
            if dim == 1:
                h = h.unsqueeze(0)
            h = h.repeat(self.ensemble_size, 1, 1)

        # standard feedforward network
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, output_size)
        return output

    def sample(self, input):
        preds = self.forward(input)

        inds = torch.randint(0, len(self.elites), input.shape[:-1])
        inds = inds.unsqueeze(dim=-1).to(device=ptu.device)
        inds = inds.repeat(1, preds.shape[2])

        samples = (inds == 0).float() * preds[self.elites[0]]
        for i in range(1, len(self.elites)):
            samples += (inds == i).float() * preds[self.elites[i]]

        return samples

    def fit_input_stats(self, data, mask=None):
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        if mask is not None:
            mean *= mask
            std *= mask

        self.input_mu.data = ptu.from_numpy(mean)
        self.input_std.data = ptu.from_numpy(std)


class ParallelizedLayerMLP(nn.Module):

    def __init__(
        self,
        ensemble_size,
        input_dim,
        output_dim,
        w_std_value=1.0,
        b_init_value=0.0
    ):
        super().__init__()

        # approximation to truncated normal of 2 stds
        w_init = ptu.randn((ensemble_size, input_dim, output_dim))
        w_init = torch.fmod(w_init, 2) * w_std_value
        self.W = nn.Parameter(w_init, requires_grad=True)

        # constant initialization
        b_init = ptu.zeros((ensemble_size, 1, output_dim)).float()
        b_init += b_init_value
        self.b = nn.Parameter(b_init, requires_grad=True)

    def forward(self, x):
        # assumes x is 3D: (ensemble_size, batch_size, dimension)
        return x @ self.W + self.b


class ParallelizedEnsembleFlattenMLP(nn.Module):

    def __init__(
            self,
            ensemble_size,
            ensemble_idx,
            win_step,
            hidden_sizes,
            input_size,
            output_size,
            init_w=3e-3,
            hidden_init=ptu.fanin_init,
            w_scale=1,
            b_init_value=0.1,
            layer_norm=None,
            batch_norm=False,
            final_init_scale=None,
    ):
        super().__init__()

        self.ensemble_size = ensemble_size
        # self.num_mins_to_use = num_mins_to_use
        self.input_size = input_size
        self.output_size = output_size
        self.elites = [i for i in range(self.ensemble_size)]

        self.sampler = np.random.default_rng()

        self.hidden_activation = F.relu
        self.output_activation = identity
        
        self.layer_norm = layer_norm

        self.fcs = []
        self.it = 0
        self.count_list = []
        self.window = ensemble_idx
        self.win_step = win_step

        # print(next(self.window))
        # exit()
        if batch_norm:
            raise NotImplementedError

        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = ParallelizedLayerMLP(
                ensemble_size=ensemble_size,
                input_dim=in_size,
                output_dim=next_size,
            )
            for j in self.elites:
                hidden_init(fc.W[j], w_scale)
                fc.b[j].data.fill_(b_init_value)
            self.__setattr__('fc%d'% i, fc)
            self.fcs.append(fc)
            in_size = next_size

        self.last_fc = ParallelizedLayerMLP(
            ensemble_size=ensemble_size,
            input_dim=in_size,
            output_dim=output_size,
        )
        if final_init_scale is None:
            self.last_fc.W.data.uniform_(-init_w, init_w)
            self.last_fc.b.data.uniform_(-init_w, init_w)
        else:
            for j in self.elites:
                ptu.orthogonal_init(self.last_fc.W[j], final_init_scale)
                self.last_fc.b[j].data.fill_(0)

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        slide_updata = kwargs.get('slide_upd', None)
        slide = kwargs.get('slide', None)
        
        
        state_dim = inputs[0].shape[-1]
        
        dim=len(flat_inputs.shape)
        # repeat h to make amenable to parallelization
        # if dim = 3, then we probably already did this somewhere else
        # (e.g. bootstrapping in training optimization)
        if dim < 3:
            flat_inputs = flat_inputs.unsqueeze(0)
            if dim == 1:
                flat_inputs = flat_inputs.unsqueeze(0)
            flat_inputs = flat_inputs.repeat(self.ensemble_size, 1, 1)
        
        # input normalization
        h = flat_inputs

        # standard feedforward network
        for _, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
            if hasattr(self, 'layer_norm') and (self.layer_norm is not None):
                h = self.layer_norm(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)

        # if original dim was 1D, squeeze the extra created layer
        if dim == 1:
            output = output.squeeze(1)

        # output is (ensemble_size, batch_size, output_size)
        # return output

        q_samples_list = []
        # window = next(self.window)

        for sample_idx in self.window:
            q_value_samples =  output[sample_idx]
            q_samples_list.append(q_value_samples)
        q_samples_list = torch.stack(q_samples_list)
        # print("111",self.ensemble_idx)
        window = copy.deepcopy(self.window)
        # print("window", window)
        if slide == 'back':
            # print('back')
            self.window = self.updata_window(self.window, self.ensemble_size, self.win_step)
            # print("self.window", self.window)
        if slide_updata:
            self.window = self.updata_window(self.window, self.ensemble_size, self.win_step)
            # print(self.window)
            # exit()
        return q_samples_list, window


    def sample(self, *inputs, slide=None):
        # self.it+=1
        
        if slide == 'back':
            slide_upd = True
        else:
            slide_upd = False
        preds, ensemble_idx = self.forward(*inputs, slide_upd=None, slide=slide)
        # print("111",ensemble_idx[:2])
        # print(preds[:2].shape)
        # exit()
        # print("torch.min(preds[-2:], dim=0)",torch.mean(torch.min(preds[-2:], dim=0)[0]))
        # print("torch.min(preds[-2:], dim=0)",torch.mean(torch.max(preds[-2:], dim=0)[0]))
        sli_f = len(ensemble_idx) - self.win_step
        sli_b = self.win_step
        # print("sli_f", sli_f)
        # print("sli_b", sli_b)
        # slib = len(ensemble_idx) - sli
        # # print("sli", -sli)
        # A = torch.min(preds[-sli:], dim=0)[0]
        # B = torch.max(preds[:slib], dim=0)[0]
        # count = (A > B).sum().item()
        # self.count_list.append(count)
        # # exit()
        # # print("count", count)
        # if self.it %1000==0:
        #     print("self.count_list", self.count_list)
        #     self.count_list = []
        # if slide == 'front':
        #     return torch.min(preds[:sli_f], dim=0)[0]
        # elif slide == 'back':
        #     return torch.min(preds[-sli_b:], dim=0)[0]
        # else:
        #     raise NotImplementedError
        # # return torch.min(preds[:2], dim=0)[0]
        return torch.min(preds[-sli_b:], dim=0)[0]

    def fit_input_stats(self, data, mask=None):
        raise NotImplementedError
    

    def circular_sliding_window(self, x: int, num: int):
        data = np.arange(x)
        step = x + 1 - num
        start = 0
        while True:
            indices = [(start + i) % x for i in range(num)]
            yield data[indices]
            start = (start + step) % x

    def updata_window(self, window, length, step):
        new_window = (window + step) % length
        return new_window
