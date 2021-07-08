import torch
import torch.nn as nn
from enum import IntEnum
import sys, os
import math, random
import itertools
import rfutils
from torch.utils.tensorboard import SummaryWriter
import numpy as np

INF = float('inf')
LOG2 = math.log(2)

PADDING = '<!PAD!>'
EOS = '<!EOS!>'

PADDING_IDX = 0
EOS_IDX = 1
DEVICE = "cuda:0"

flat = itertools.chain.from_iterable


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class Gaussian_Sampler(nn.Module):

    """
    Module to sample from multivariate Gaussian distributions.
    Converts input into a sampled vector of the same size.
    """

    def __init__(self, input_size: int):
        super(Gaussian_Sampler, self).__init__()
        self.input_size = input_size
        self.gauss_parameter_generator = nn.Parameter(torch.FloatTensor(self.input_size, self.input_size*2))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def sample(self, stats):
        mu = stats[:, :, :self.input_size].clamp(min=1e-6, max=1e6)
        std = nn.functional.softplus(stats[:, :, self.input_size:], beta=1).clamp(min=1e-6)
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        # output noisy sampling of size B x T x H
        return mu, std, mu + eps * std

    def forward(self, x: torch.Tensor):
        # x is of size TxBxF
        gauss_parameters = x @ self.gauss_parameter_generator
        mu, std, sample = self.sample(gauss_parameters)

        return (sample, mu, std)


class Recursive_Gaussian_LSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        super(Recursive_Gaussian_LSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz

        self.lstm = torch.nn.LSTM(
            input_size=self.input_sz,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )


    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    # def forward(self, x: torch.Tensor,
    #             init_states=None):
    #     """Assumes x is of shape (batch, sequence, feature)"""
    #     bs, seq_sz, _ = x.size()
    #     h_distributions = []
    #     c_distributions = []
    #     if init_states is None:
    #         h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
    #                     torch.zeros(self.hidden_size).to(x.device))
    #     else:
    #         h_t, c_t = init_states
    #
    #     hidden_seq = []
    #     c_seq = []
    #
    #     HS = self.hidden_size
    #     for t in range(seq_sz):
    #         x_t = x[:, t, :]
    #         output, (out_h, out_c) = self.lstm(x_t, (h_t, c_t))
    #         # batch the computations into a single matrix multiplication
    #         gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
    #         i_t, f_t, g_t, o_t = (
    #             torch.sigmoid(gates[:, :HS]),  # input
    #             torch.sigmoid(gates[:, HS:HS * 2]),  # forget
    #             torch.tanh(gates[:, HS * 2:HS * 3]),
    #             torch.sigmoid(gates[:, HS * 3:]),  # output
    #         )
    #         pre_c_t = f_t * c_t + i_t * g_t
    #         pre_h_t = o_t * torch.tanh(c_t)
    #
    #         # Produce parameters for Gaussian distribution of hidden_size dimensions
    #         # sample from Gaussian distribution
    #         # otherwise, pass pre_c_t/pre_h_t on as c_t/h_t itself
    #         if self.noise == ("h" or "both"):
    #             h_params = pre_h_t @ self.h_reparameterize
    #             h_mu, h_std, h_t = self.sample(h_params)
    #             h_stats.append(torch.cat([h_mu, h_std], dim=Dim.seq).unsqueeze(Dim.batch))
    #         else:
    #             h_t = pre_h_t
    #         if self.noise == ("c" or "both"):
    #             c_params = pre_c_t @ self.c_reparameterize
    #             c_mu, c_std, c_t = self.sample(c_params)
    #             c_stats.append(torch.cat([c_mu, c_std], dim=Dim.seq).unsqueeze(Dim.batch))
    #         else:
    #             c_t = pre_c_t
    #
    #         # append returnable sequences
    #         c_seq.append(c_t.unsqueeze(Dim.batch))
    #         hidden_seq.append(h_t.unsqueeze(Dim.batch))
    #
    #         # c_stats.append(torch.cat([c_mu, c_std],dim=Dim.seq).unsqueeze(Dim.batch))
    #         # h_stats.append(torch.cat([h_mu, h_std],dim=Dim.seq).unsqueeze(Dim.batch))
    #
    #     hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
    #     c_seq = torch.cat(c_seq, dim=Dim.batch)
    #
    #     # Improvable later, but this section transforms and splits the stats tensors if relevant
    #     # Tensor shape B x T x H
    #     if self.noise == ("h" or "both"):
    #         h_stats = torch.cat(h_stats, dim=Dim.batch)
    #         h_stats = h_stats.transpose(Dim.batch, Dim.seq).contiguous()
    #         h_stds = h_stats[:, :, self.hidden_size:].contiguous()
    #         h_mus = h_stats[:, :, :self.hidden_size].contiguous()
    #     else:
    #         h_mus = None
    #         h_stds = None
    #     if self.noise == ("c" or "both"):
    #         c_stats = torch.cat(c_stats, dim=Dim.batch)
    #         c_stats = c_stats.transpose(Dim.batch, Dim.seq).contiguous()
    #         c_stds = c_stats[:, :, self.hidden_size:].contiguous()
    #         c_mus = c_stats[:, :, :self.hidden_size].contiguous()
    #     else:
    #         c_mus = None
    #         c_stds = None
    #
    #     # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
    #     hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
    #     c_seq = c_seq.transpose(Dim.batch, Dim.seq).contiguous()
    #     # h_stats = h_stats.transpose(Dim.batch, Dim.seq).contiguous()
    #     # c_stats = c_stats.transpose(Dim.batch, Dim.seq).contiguous()
    #     return hidden_seq, c_seq, (h_mus, h_stds), (c_mus, c_stds), (h_t, c_t)

    # def sample(self, stats):
    #     mu = stats[:, :self.hidden_size].clamp(min=1e-6, max=1e6)
    #     std = nn.functional.softplus(stats[:, self.hidden_size:], beta=1).clamp(min=1e-6)
    #     eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    #     # output noisy sampling of size B x T x H
    #     return mu, std, mu + eps * std


class h_RGLSTM(Recursive_Gaussian_LSTM):
    def __init__(self, input_sz: int, hidden_sz: int):
        super(h_RGLSTM, self).__init__(input_sz, hidden_sz)
        self.h_gauss_sampler = Gaussian_Sampler(hidden_sz)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        h_mus = []
        h_stds = []
        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.hidden_size).to(x.device)] * bs, dim=1),
                        torch.stack([torch.zeros(self.hidden_size).to(x.device)] * bs, dim=1))
        else:
            h_t, c_t = init_states

        hidden_seq = []
        h_t = h_t.unsqueeze(dim=0)
        c_t = c_t.unsqueeze(dim=0)


        h_t, h_mu, h_std = self.h_gauss_sampler(h_t)
        h_mus.append(h_mu.squeeze(dim=0))
        h_stds.append(h_std.squeeze(dim=0))
        #c_seq = []

        #HS = self.hidden_size

        for t in range(seq_sz):
            x_t = x[:, t, :]
            output, (out_h, out_c) = self.lstm(x_t.unsqueeze(dim=1), (h_t, c_t))
            c_t = out_c
            h_t, h_mu, h_std = self.h_gauss_sampler(out_h)
            h_mus.append(h_mu.squeeze(dim=0))
            h_stds.append(h_std.squeeze(dim=0))
            #h_distributions.append(torch.cat([h_mu, h_std], dim=Dim.seq).unsqueeze(Dim.batch))
            hidden_seq.append(h_t)


        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch).squeeze()
        h_mus = torch.stack(h_mus).transpose(Dim.batch, Dim.seq).contiguous()
        h_stds = torch.stack(h_stds).transpose(Dim.batch, Dim.seq).contiguous()
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, ((h_mus,h_stds), None), (h_t, c_t)


class c_RGLSTM(Recursive_Gaussian_LSTM):

    def __init__(self, input_sz: int, hidden_sz: int):
        super(c_RGLSTM, self).__init__(input_sz, hidden_sz)
        self.c_gauss_sampler = Gaussian_Sampler(hidden_sz)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        c_mus = []
        c_stds = []
        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.hidden_size).to(x.device)] * bs, dim=-2),
                        torch.stack([torch.zeros(self.hidden_size).to(x.device)] * bs, dim=-2))
        else:
            h_t, c_t = init_states

        hidden_seq = []
        h_t = h_t.unsqueeze(dim=0)
        c_t = c_t.unsqueeze(dim=0)

        c_t, c_mu, c_std = self.h_gauss_sampler(c_t)
        c_mus.append(c_mu.squeeze(dim=0))
        c_stds.append(c_std.squeeze(dim=0))

        for t in range(seq_sz):
            x_t = x[:, t, :]
            output, (out_h, out_c) = self.lstm(x_t.unsqueeze(dim=1), (h_t, c_t))
            h_t = out_h
            c_t, c_mu, c_std = self.c_gauss_sampler(out_c)
            c_mus.append(c_mu.squeeze(dim=0))
            c_stds.append(c_std.squeeze(dim=0))
            hidden_seq.append(h_t)


        hidden_seq = torch.cat(hidden_seq)
        c_mus = torch.stack(c_mus).transpose(Dim.batch, Dim.seq).contiguous()
        c_stds = torch.stack(c_stds).transpose(Dim.batch, Dim.seq).contiguous()
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (None, (c_mus,c_stds)), (h_t, c_t)


class b_RGLSTM(Recursive_Gaussian_LSTM):

    def __init__(self, input_sz: int, hidden_sz: int):
        super(b_RGLSTM, self).__init__(input_sz, hidden_sz)
        self.h_gauss_sampler = Gaussian_Sampler(hidden_sz)
        self.c_gauss_sampler = Gaussian_Sampler(hidden_sz)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        h_mus = []
        h_stds = []
        c_mus = []
        c_stds = []

        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.hidden_size).to(x.device)]*bs, dim=1),
                        torch.stack([torch.zeros(self.hidden_size).to(x.device)]*bs, dim=1))
        else:
            h_t, c_t = init_states

        hidden_seq = []
        h_t = h_t.unsqueeze(dim=0)
        c_t = c_t.unsqueeze(dim=0)

        #print(h_t.size())

        c_t, c_mu, c_std = self.h_gauss_sampler(c_t)
        c_mus.append(c_mu.squeeze(dim=0))
        c_stds.append(c_std.squeeze(dim=0))
        h_t, h_mu, h_std = self.h_gauss_sampler(h_t)
        h_mus.append(h_mu.squeeze(dim=0))
        h_stds.append(h_std.squeeze(dim=0))

        for t in range(seq_sz):
            x_t = x[:, t, :]
            output, (out_h, out_c) = self.lstm(x_t.unsqueeze(dim=1), (h_t, c_t))
            h_t, h_mu, h_std = self.h_gauss_sampler(out_h)
            c_t, c_mu, c_std = self.c_gauss_sampler(out_c)
            h_mus.append(h_mu.squeeze(dim=0))
            h_stds.append(h_std.squeeze(dim=0))
            c_mus.append(c_mu.squeeze(dim=0))
            c_stds.append(c_std.squeeze(dim=0))
            hidden_seq.append(h_t)

        hidden_seq = torch.cat(hidden_seq)
        h_mus = torch.stack(h_mus).transpose(Dim.batch, Dim.seq).contiguous()
        h_stds = torch.stack(h_stds).transpose(Dim.batch, Dim.seq).contiguous()
        c_mus = torch.stack(c_mus).transpose(Dim.batch, Dim.seq).contiguous()
        c_stds = torch.stack(c_stds).transpose(Dim.batch, Dim.seq).contiguous()
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, ((h_mus, h_stds), (c_mus, c_stds)), (h_t, c_t)


class Blanket_Gaussian_LSTM(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, num_layers: int):
        super(Blanket_Gaussian_LSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            input_size=self.input_sz,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )

        self.h_gauss_sampler = Gaussian_Sampler(hidden_sz)

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor,
                init_states = None):

        bs, seq_sz, _ = x.size()

        if init_states is None:
            h_t, c_t = (torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2),
                        torch.stack([torch.zeros(self.num_layers, self.hidden_size).to(x.device)] * bs, dim=-2))
        else:
            h_t, c_t = init_states
        #h_t = h_t.unsqueeze(dim=0)
        #c_t = c_t.unsqueeze(dim=0)
        output, (out_h, out_c) = self.lstm(x, (h_t, c_t))
        h_seq, h_mus, h_stds = self.h_gauss_sampler(torch.cat((torch.transpose(h_t,0,1), output), dim=-2)[:, 0:-1, :])

        return h_seq, (h_mus, h_stds)

class Recursive_Gauss_LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size, noise, va_z=True,
                 padding_idx=PADDING_IDX, eos_idx=EOS_IDX):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.noise = noise
        self.va_z = va_z

        self.__build_model()

    def get_Z_h(self):
        std = nn.functional.softplus(self.Z_h, beta=1).clamp(min=1e-6)
        return std

    def get_Z_c(self):
        std = nn.functional.softplus(self.Z_c, beta=1).clamp(min=1e-6)
        return std


    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        # For the creation of multi-layer LSTMs
        if self.noise == "h":
            g = [h_RGLSTM(self.embedding_dim, self.hidden_size)]
            g.extend(
                [h_RGLSTM(self.hidden_size, self.hidden_size) for _ in range(1, self.num_layers)])
        if self.noise == "c":
            g = [c_RGLSTM(self.embedding_dim, self.hidden_size)]
            g.extend(
                [c_RGLSTM(self.hidden_size, self.hidden_size) for _ in range(1, self.num_layers)])
        if self.noise == "both":
            g = [b_RGLSTM(self.embedding_dim, self.hidden_size)]
            g.extend(
                [b_RGLSTM(self.hidden_size, self.hidden_size) for _ in range(1, self.num_layers)])

        if self.va_z:
            #Initialize cov matrix to be identity matrix
            if self.noise == "h" or self.noise == "both":
                Z_h = torch.ones(self.num_layers, self.hidden_size)
                self.Z_h = torch.nn.Parameter(Z_h)

            if self.noise == "c" or self.noise == "both":
                Z_c = torch.ones(self.num_layers, self.hidden_size)
                self.Z_c = torch.nn.Parameter(Z_c)



        self.rglstm_layers = nn.ModuleList(g)

        self.initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_size),
             torch.randn(self.num_layers, self.hidden_size)]
        ))

        self.decoder = torch.nn.Linear(self.hidden_size, self.vocab_size)

    def get_initial_state(self, batch_size):
        init_a, init_b = self.initial_state
        return torch.stack([init_a] * batch_size, dim=-2), torch.stack([init_b] * batch_size, dim=-2)  # L x B x H

    def encode(self, X):
        # X is of shape B x T
        # To add: logic to make flag choice of "c"/"h" better
        batch_size, seq_len = X.shape
        embedding = self.word_embedding(X)
        init_h, init_c = self.get_initial_state(batch_size)  # init_h is L x B x H
        c_mus = []
        h_mus = []
        h_stds = []
        c_stds = []
        seq = embedding
        for i, layer in enumerate(self.rglstm_layers):
            seq, ((h_mu, h_std), (c_mu, c_std)), (_, __) = layer(seq, (init_h[i], init_c[i]))
            c_mus.append(c_mu)
            c_stds.append(c_std)
            h_mus.append(h_mu)
            h_stds.append(h_std)

        # If noise flag "h"/"c", transforms them, otherwise don't touch because it's a NoneType

        if self.noise == "c" or self.noise == "both":
            c_mus = torch.cat(c_mus, Dim.feature)[:, 0:-1, :]
            c_stds = torch.cat(c_stds, Dim.feature)[:, 0:-1, :]

        if self.noise == "h" or self.noise == "both":
            h_mus = torch.cat(h_mus, Dim.feature)[:, 0:-1, :]
            h_stds = torch.cat(h_stds, Dim.feature)[:, 0:-1, :]

        c_dists = (c_mus, c_stds)
        h_dists = (h_mus, h_stds)

        # Also need to include the initial state itself
        # And the last state is after consuming EOS, so it doesn't matter
        # So add the initial state in first, and remove the final state
        seq = torch.cat((init_h[-1].unsqueeze(-2), seq), dim=-2)[:, 0:-1, :]

        return seq, (h_dists, c_dists)

    def decode(self, h):
        logits = self.decoder(h)
        # Remove probability mass from the pad token
        logits[:, :, self.padding_idx] = -INF
        return torch.log_softmax(logits, -1)

    def lm_loss(self, y, y_hat):
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions
        y_flat = y.view(-1)  # flatten to B*T
        y_hat_flat = y_hat.view(-1, self.vocab_size)  # flatten to B*T x V
        mask = (y_flat == self.padding_idx)
        num_tokens = torch.sum(~mask).item()
        y_hat_correct = y_hat_flat[range(y_hat_flat.shape[0]), y_flat]
        y_hat_correct[mask] = 0
        ce_loss = -torch.sum(y_hat_correct) / num_tokens
        return ce_loss

    def mi_loss(self, stats, y, dist):
        if self.va_z:
            if dist == "h":
                std_z = self.get_Z_h()
            else:
                std_z = self.get_Z_c()
            mu_z = torch.zeros(self.hidden_size).to("cuda")
            std = stats[1]
            mu = stats[0]
            mask = (y != self.padding_idx)
            std_flat = std.contiguous()[mask, :]
            mu_flat = mu.contiguous()[mask, :]
            mi_loss = kl_divergence(mu_flat, std_flat, mu_z, std_z)

        else:
            std = stats[1]  # stats[:,:,self.hidden_size:].contiguous()
            mu = stats[0]  # [:, :, :self.hidden_size].contiguous()
            mask = (y != self.padding_idx)
            #n = mask.sum()

            std_flat = std.contiguous()[mask, :]  # B*TxH remove padded entries
            mu_flat = mu.contiguous()[mask, :]  # B*TxH, ditto

            mi_loss = (-(1 / 2) * (std_flat.log().sum(dim=1) + self.hidden_size -
                                   (mu_flat ** 2).sum(dim=1) - std_flat.sum(dim=1)))

        return mi_loss.mean()

    def pmi(self, stats):
        std = stats[1]
        mu = stats[0]
        d = len(mu)
        pmi = (1/2)*(std.sum()-d-std.log().sum()+(mu**2).sum())

        return pmi

    def process_data(self, data, batch_size, alpha, beta, validation=False):
        m = sum([len(word) for word in data])
        data_copy = data.copy()
        random.shuffle(data_copy)
        batches = split_list(data_copy, batch_size)
        avg_loss = torch.FloatTensor([0]).to("cuda")
        avg_hib_loss = torch.FloatTensor([0]).to("cuda")
        avg_cib_loss = torch.FloatTensor([0]).to("cuda")
        avg_ce_loss = torch.FloatTensor([0]).to("cuda")
        # with torch.autograd.detect_anomaly():
        with torch.set_grad_enabled(not validation):
            for batch in batches:
                self.opt.zero_grad()
                n = sum([len(word) for word in batch])
                w = n / m
                padded_batch = pad_sequences(batch)
                padded_batch = padded_batch.to("cuda")
                hidden_seq, (h_dists, c_dists) = self.encode(padded_batch)
                y_hat = self.decode(hidden_seq)  # shape B x (T+1) x V ???
                if self.noise == "h" or self.noise == "both":
                    hib_loss = self.mi_loss(h_dists, padded_batch, "h")
                else:
                    hib_loss = 0
                if self.noise == "c" or self.noise == "both":
                    cib_loss = self.mi_loss(c_dists, padded_batch, "c")
                else:
                    cib_loss = 0
                ce_loss = self.lm_loss(padded_batch, y_hat)
                loss = alpha * hib_loss + beta * cib_loss + ce_loss
                avg_loss += loss * w
                avg_hib_loss += hib_loss * w
                avg_cib_loss += cib_loss * w
                avg_ce_loss += ce_loss * w
                # yield (loss, avg_loss, avg_hib_loss, avg_cib_loss, avg_ce_loss)

                if not validation:
                    #     print("loss " + str(loss.isnan().any()))
                    loss.backward()
                    self.opt.step()

            del padded_batch
        return loss, ce_loss, avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss

    def train_lm(self, data, print_every=10, num_epochs=1000, early_stopping=True, batch_size=None, alpha_start=0,
                 alpha_end=.01, beta_start=0, beta_end=0, annealing = "none", **kwds):
        writer = SummaryWriter()
        data_size = len(data)
        hold_out_size = math.floor(data_size * .15)
        random.shuffle(data)
        testing_data = data[-hold_out_size:]
        training_data = data[:-hold_out_size]
        #self.Z.requires_grad = False
        #self.Z_on = False
        if batch_size is None:
            batch_size = len(data)
        self.opt = torch.optim.Adam(lr=.01, params=self.parameters(), **kwds)

        #institutes generators for the "lambda" parameters (beta and alpha) used during training
        if annealing.lower() == "none":
            alpha = itertools.repeat(alpha_end)
            beta = itertools.repeat(beta_end)
        elif annealing.lower() == "linear":
            alpha = linear_annealing(alpha_start, alpha_end, num_epochs)
            beta = linear_annealing(beta_start, beta_end, num_epochs)
            if early_stopping != False:
                print("For linear annealing cannot implement early stopping, early stopping automatically turned off.")
                early_stopping = False
        elif annealing.lower() == "elbow":
            alpha = elbow_annealing(alpha_start, alpha_end, num_epochs, .75)
            beta = elbow_annealing(beta_start, beta_end, num_epochs, .75)
        elif annealing.lower() == "double_elbow":
            alpha = double_elbow_annealing(alpha_start, alpha_end, num_epochs, .25, .75)
            beta = double_elbow_annealing(beta_start, beta_end, num_epochs, .25, .75)
        else:
            print("%s is not an annealing type, defaulting to no annealing." % annealing)
            alpha = itertools.repeat(alpha_end)
            beta = itertools.repeat(beta_end)

        training_loss = []
        testing_loss = []
        self.to("cuda")

        #used for early stopping
        best_loss = float("Inf")
        best_ce_loss = float("Inf")
        best_hib_loss = float("Inf")
        best_cib_loss = float("Inf")
        epochs_no_improvement = 0
        try:
            for i in range(num_epochs):
                a = next(alpha)
                b = next(beta)
                loss, ce_loss, avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss = self.process_data(training_data,
                                                                                                     batch_size, a, b)
                training_loss.append(avg_loss)
                writer.add_scalar("Loss/train", loss.item() / LOG2, i)
                writer.add_scalar("CE/train", ce_loss.item() / LOG2, i)

                if i % print_every == 0:
                    print(
                        "epoch %d, loss = %s, cross entropy = %s, h mutual information = %s, c mutual information = %s" % (
                            i, str(avg_loss.item() / LOG2), str(avg_ce_loss.item() / LOG2),
                            str(avg_hib_loss.item() / LOG2),
                            str(avg_cib_loss.item() / LOG2)), file=sys.stderr)
                    loss, ce_loss, avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss = self.process_data(testing_data,
                                                                                                         batch_size, a,
                                                                                                         b,
                                                                                                         validation=True)
                    print(
                        "testing loss = %s, cross entropy = %s, h mutual information = %s, c mutual information = %s" % (
                            str(avg_loss.item() / LOG2), str(avg_ce_loss.item() / LOG2),
                            str(avg_hib_loss.item() / LOG2),
                            str(avg_cib_loss.item() / LOG2)), file=sys.stderr)
                    writer.add_scalar("Loss/test", avg_loss.item() / LOG2, i)
                    writer.add_scalar("CE/test", avg_ce_loss.item() / LOG2, i)
                    testing_loss.append(loss)

                #Implementing early stopping below
                #Must be past elbow during annealing
                if early_stopping:
                    #.75 below comes from the elbows in the elbow annealers, currently not a passable parameter
                    #COMING SOON
                    if annealing == "none" or (i>=.75*num_epochs):
                        if avg_loss.item() < best_loss:
                            epochs_no_improvement = 0
                            torch.save(self.state_dict(), "cache/temp_best_model.pt")
                            best_loss = avg_loss.item()
                            best_loss_t = avg_loss
                            best_ce_loss = avg_ce_loss
                            best_cib_loss = avg_cib_loss
                            best_hib_loss = avg_hib_loss
                        else:
                            epochs_no_improvement += 1
                            if epochs_no_improvement > 4:
                                #if not self.Z_on:
                                #    print("Updating Z")
                                #    self.Z.requires_grad = True
                                #    self.Z_on = True
                                #    epochs_no_improvement = 0
                                #else:
                                print("Early Stopping")
                                self.load_state_dict(torch.load("cache/temp_best_model.pt"))
                                return best_loss_t, best_ce_loss, best_hib_loss, best_cib_loss
        except KeyboardInterrupt:
            print("Training interrupted.")
            pass
        writer.flush()
        writer.close()

        return avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss

        #return (avg_hib_loss / avg_cib_loss)

    def distro_after(self, sequence):
        sequence = list(sequence) + [PADDING_IDX]
        padded = pad_sequences([sequence])
        hidden_seq, (_, __) = self.encode(padded)
        predicted = self.decode(hidden_seq)[0, -1, :]
        return predicted

    def generate(self):
        so_far = []
        while True:
            predicted = self.distro_after(so_far).exp().detach().numpy()
            sampled = np.random.choice(range(len(predicted)), p=predicted)
            yield sampled
            if sampled == self.eos_idx:
                break
            else:
                so_far.append(sampled)

    def mi_read(self, word):
        h_pmi = []
        c_pmi = []
        hidden_seq, c_seq, h_stats, c_stats = self.encode(pad_sequences([word]))
        (h_mus, h_stds) = h_stats
        (c_mus, c_stds) = c_stats
        if self.noise == "h" or self.noise == "both":
            for i, mu in enumerate(torch.squeeze(h_mus)):
                print(mu)
                print(torch.squeeze(h_stds)[i])
                stats = (mu, torch.squeeze(h_stds)[i])
                h_pmi.append(self.pmi(stats))
        if self.noise == "c" or self.noise == "both":
            for i, mu in enumerate(torch.squeeze(c_mus)):
                stats = (mu, torch.squeeze(c_stds)[i])
                c_pmi.append(self.pmi(stats))
        #     for i, (a, b) in enumerate(rfutils.sliding(torch.squeeze(h_mus), 2)):
        #         h_pmi = self.pmi(a, torch.squeeze(h_stds)[i], b, torch.squeeze(h_stds)[i + 1])
        #         print(kld)
        #         h_kl_seq.append(kld)
        # if self.noise == "h" or self.noise == "both":
        #     for i, (a, b) in enumerate(rfutils.sliding(torch.squeeze(h_mus), 2)):
        #         kld = kl_divergence(a, torch.squeeze(c_stds)[i], b, torch.squeeze(h_stds)[i + 1])
        #         print(kld)
        #         c_kl_seq.append(kld)
        return h_pmi, c_pmi

    def sample(self, stats):
        mu = stats[:, :self.hidden_size].clamp(min=1e-6, max=1e6)
        std = nn.functional.softplus(stats[:, self.hidden_size:], beta=1).clamp(min=1e-6)
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        # output noisy sampling of size B x T x H
        return mu, std, mu + eps * std

class Blanket_Gauss_LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size,
                 padding_idx=PADDING_IDX, eos_idx=EOS_IDX):

        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size*2
        self.noise_size = hidden_size
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx

        self.__build_model()

    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )

        self.lstm = Blanket_Gaussian_LSTM(self.embedding_dim, self.hidden_size, self.num_layers)

        self.initial_state = torch.nn.Parameter(torch.stack(
            [torch.randn(self.num_layers, self.hidden_size),
             torch.randn(self.num_layers, self.hidden_size)]
        ))

        self.decoder = torch.nn.Linear(self.noise_size, self.vocab_size)

    def get_initial_state(self, batch_size):
        init_a, init_b = self.initial_state
        return torch.stack([init_a] * batch_size, dim=-2), torch.stack([init_b] * batch_size, dim=-2)  # L x B x H

    def encode(self, X):
        # X is of shape B x T
        batch_size, seq_len = X.shape
        embedding = self.word_embedding(X)
        init_h, init_c = self.get_initial_state(batch_size)  # init_h is L x B x H
        h_seq, (h_mus, h_stds) = self.lstm(embedding, (init_h, init_c))  # output is B x T x H
        # Also need to include the initial state itself
        # And the last state is after consuming EOS, so it doesn't matter
        # So add the initial state in first, and remove the final state
        return h_seq, (h_mus, h_stds)

    def decode(self, h):
        logits = self.decoder(h)
        # Remove probability mass from the pad token
        logits[:, :, self.padding_idx] = -INF
        return torch.log_softmax(logits, -1)

    def process_data(self, data, batch_size, alpha, beta, validation=False):
        m = sum([len(word) for word in data])
        data_copy = data.copy()
        random.shuffle(data_copy)
        batches = split_list(data_copy, batch_size)
        avg_loss = torch.FloatTensor([0]).to("cuda")
        avg_ib_loss = torch.FloatTensor([0]).to("cuda")
        avg_ce_loss = torch.FloatTensor([0]).to("cuda")
        # with torch.autograd.detect_anomaly():
        with torch.set_grad_enabled(not validation):
            for batch in batches:
                self.opt.zero_grad()
                n = sum([len(word) for word in batch])
                w = n / m
                padded_batch = pad_sequences(batch)
                padded_batch = padded_batch.to("cuda")
                h_seq, h_stats = self.encode(padded_batch)
                y_hat = self.decode(h_seq)  # shape B x (T+1) x V ???
                ce_loss = self.lm_loss(padded_batch, y_hat)
                ib_loss = self.mi_loss(h_stats, y_hat, "h")
                loss = beta * ib_loss + ce_loss
                avg_loss += loss * w
                avg_ce_loss += ce_loss * w
                avg_ib_loss += ib_loss * w
                # yield (loss, avg_loss, ib_loss, avg_ce_loss)

                if not validation:
                    #     print("loss " + str(loss.isnan().any()))
                    loss.backward()
                    self.opt.step()

            del padded_batch
        return loss, ce_loss, ib_loss, avg_loss, avg_ib_loss, avg_ce_loss

    def train_lm(self, data, print_every=10, num_epochs=1000, batch_size=None, alpha_start=0, alpha_end=.01,
                 beta_start=0, beta_end=0, **kwds):
        writer = SummaryWriter()
        data_size = len(data)
        hold_out_size = math.floor(data_size * .15)
        random.shuffle(data)
        testing_data = data[-hold_out_size:]
        training_data = data[:-hold_out_size]
        if batch_size is None:
            batch_size = len(data)
        self.opt = torch.optim.Adam(lr=.01, params=self.parameters(), **kwds)
        alpha = elbow_annealing(alpha_start, alpha_end, num_epochs, .75)
        beta = elbow_annealing(beta_start, beta_end, num_epochs, .75)
        training_loss = []
        testing_loss = []
        self.to("cuda")
        try:
            for i in range(num_epochs):
                a = next(alpha)
                b = next(beta)
                loss, ce_loss, avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss = self.process_data(training_data,
                                                                                                     batch_size, a, b)
                training_loss.append(avg_loss)
                writer.add_scalar("Loss/train", loss.item() / LOG2, i)
                writer.add_scalar("CE/train", ce_loss.item() / LOG2, i)

                if i % print_every == 0:
                    print(
                        "epoch %d, loss = %s, cross entropy = %s, h mutual information = %s, c mutual information = %s" % (
                            i, str(avg_loss.item() / LOG2), str(avg_ce_loss.item() / LOG2),
                            str(avg_hib_loss.item() / LOG2),
                            str(avg_cib_loss.item() / LOG2)), file=sys.stderr)
                    loss, ce_loss, avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss = self.process_data(testing_data,
                                                                                                         batch_size, a,
                                                                                                         b,
                                                                                                         validation=True)
                    print(
                        "testing loss = %s, cross entropy = %s, h mutual information = %s, c mutual information = %s" % (
                            str(avg_loss.item() / LOG2), str(avg_ce_loss.item() / LOG2),
                            str(avg_hib_loss.item() / LOG2),
                            str(avg_cib_loss.item() / LOG2)), file=sys.stderr)
                    writer.add_scalar("Loss/test", avg_loss.item() / LOG2, i)
                    writer.add_scalar("CE/test", avg_ce_loss.item() / LOG2, i)
                    testing_loss.append(loss)
        except KeyboardInterrupt:
            print("Training interrupted.")
            pass
        writer.flush()
        writer.close()

        return avg_hib_loss, avg_cib_loss

    def lm_loss(self, y, y_hat):
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions
        y_flat = y.view(-1)  # flatten to B*T
        y_hat_flat = y_hat.view(-1, self.vocab_size)  # flatten to B*T x V
        mask = (y_flat == self.padding_idx)
        num_tokens = torch.sum(~mask).item()
        y_hat_correct = y_hat_flat[range(y_hat_flat.shape[0]), y_flat]
        y_hat_correct[mask] = 0
        ce_loss = -torch.sum(y_hat_correct) / num_tokens
        return ce_loss

    def mi_loss(self, stats, y, dist):
        if self.va_z:
            if dist == "h":
                std_z = self.get_Z_h()
            else:
                std_z = self.get_Z_c()
            mu_z = torch.zeros(self.hidden_size).to("cuda")
            std = stats[1]
            mu = stats[0]
            mask = (y != self.padding_idx)
            std_flat = std.contiguous()[mask, :]
            mu_flat = mu.contiguous()[mask, :]
            mi_loss = kl_divergence(mu_flat, std_flat, mu_z, std_z)

        else:
            std = stats[1]  # stats[:,:,self.hidden_size:].contiguous()
            mu = stats[0]  # [:, :, :self.hidden_size].contiguous()
            mask = (y != self.padding_idx)
            # n = mask.sum()

            std_flat = std.contiguous()[mask, :]  # B*TxH remove padded entries
            mu_flat = mu.contiguous()[mask, :]  # B*TxH, ditto

            mi_loss = (-(1 / 2) * (std_flat.log().sum(dim=1) + self.hidden_size -
                                   (mu_flat ** 2).sum(dim=1) - std_flat.sum(dim=1)))

        return mi_loss.mean()

    #def pmi(self, stats):


    def sample(self, stats):

        mu = stats[:, :self.hidden_size].clamp(min=1e-6, max=1e6)
        std = nn.functional.softplus(stats[:, self.hidden_size:], beta=1).clamp(min=1e-6)
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        # output noisy sampling of size B x T x H
        return mu, std, mu + eps * std

def pad_sequences(xs, padding_idx=PADDING_IDX):
    batch_size = len(xs)
    lengths = [len(x) for x in xs]
    longest = max(lengths)
    padded = torch.ones(batch_size, longest).long() * padding_idx
    for i, length in enumerate(lengths):
        sequence = xs[i]
        padded[i, 0:length] = torch.Tensor(sequence[:length])
    return padded


def example(**kwds):
    """ Minimum possible loss is 1/5 ln 5 = 0.3219 """
    data = [
        [2, 2, EOS_IDX],
        [3, 3, 3, EOS_IDX],
        [4, 4, 4, 4, EOS_IDX],
        [5, 5, 5, 5, 5, EOS_IDX],
        [6, 6, 6, 6, 6, 6, EOS_IDX],
    ]
    vocab_size = len(data) + 2
    lstm = LSTM(vocab_size, vocab_size, 2, 10, "h")
    lstm.train_lm(data, num_epochs=2000, **kwds)
    return lstm


class Indexer:

    def __init__(self, eos_idx=EOS_IDX, padding_idx=PADDING_IDX):
        self.counter = itertools.count()
        self.eos_idx = eos_idx
        self.padding_idx = padding_idx
        self.seen = {EOS: self.eos_idx, PADDING: self.padding_idx}

    def index_for(self, x):
        if x in self.seen:
            return self.seen[x]
        else:
            result = next(self.counter)
            while result == self.eos_idx or result == self.padding_idx:
                result = next(self.counter)
            self.seen[x] = result
            return result

    def format_sequence(self, xs):
        result = list(map(self.index_for, xs))
        result.append(self.eos_idx)
        return result

    @property
    def vocab(self):
        rvocab = [(i, w) for w, i in self.seen.items()]
        return [w for i, w in sorted(rvocab)]


def format_sequences(xss):
    indexer = Indexer()
    result = list(map(indexer.format_sequence, xss))
    return result, indexer.vocab, indexer


def read_unimorph(filename, field=1):
    with open(filename) as infile:
        for line in infile:
            if line.strip():
                parts = line.strip().split("\t")
                yield parts[field].casefold()


# One-step Predictive Information Bottleneck objective is
# J = -I[Z:X_t] + I[X_{<t}:Z]
# This is bounded as
# J < < E_P(Z|X_{<t}) log Q(X_t|Z) + D[P(Z|X) || R(Z)] >,
# where the minimization is over P, Q, and R, and the outer expectation is over the empirical p(X_t, X_{<t})

# In Futrell & Hahn (2019), P is a Gaussian distribution whose mu and diagonal Sigma are determined by matrices W_mu and W_sigma which decode an LSTM.
# The gradient is approximated by drawing a single sample from P, and using the reparameterized gradient estimator of Kingma and Welling
#


def train_unimorph_recursive_lm(lang, hidden_size=100, num_layers=2, batch_size=2048, num_epochs=50, print_every=2,
                      embedding_dim = 20, noise="h", beta_start = .0005, beta_end= .005, annealing = "none", va = True, **kwds):
    data, vocab, indexer = list(format_sequences(read_unimorph("%s" % lang)))
    print("Loaded data for %s..." % lang, file=sys.stderr)
    vocab_size = len(vocab)
    print("Vocab size: %d" % vocab_size, file=sys.stderr)
    lstm = Recursive_Gauss_LSTM(vocab_size, embedding_dim, int(num_layers), int(hidden_size), noise=noise, va_z=va).to("cuda:0")
    # lstm.double()
    print(lstm, file=sys.stderr)
    avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss = lstm.train_lm(data, num_epochs=int(num_epochs), batch_size=int(batch_size),
                                                            print_every=int(print_every), beta_start=beta_start, beta_end=beta_end,
                                                            alpha_start=beta_start, alpha_end=beta_end, annealing = annealing, **kwds)
    lstm.to("cpu")
    num_samples = 10
    print("Generating %d samples..." % num_samples, file=sys.stderr)
    for _ in range(num_samples):
        symbols = list(lstm.generate())[:-1]
        print("".join(map(vocab.__getitem__, symbols)), file=sys.stderr)
    #lstm.mi_read(indexer.format_sequence("bending"))

    #torch.save(lstm, "char_model.lstm")
    return avg_loss, avg_ce_loss, avg_hib_loss, avg_cib_loss, lstm, vocab


def train_many_both(lang, hidden_size=100, num_layers=2, batch_size=2048, num_epochs=80, print_every=2,
                    num_samples=25, start_lambda=.0005, end_lambda=.005, step_size=.0002, **kwds):
    meta_writer = SummaryWriter()
    data, vocab, indexer = list(format_sequences(read_unimorph("%s" % lang)))
    print("Loaded data for %s..." % lang, file=sys.stderr)
    vocab_size = len(vocab)
    print("Vocab size: %d" % vocab_size, file=sys.stderr)
    lam = start_lambda
    ratios = []
    while (lam <= end_lambda):
        lstm = LSTM(vocab_size, 20, int(num_layers), int(hidden_size), noise="h").to("cuda:0")
        r = lstm.train_lm(data, num_epochs=int(num_epochs), batch_size=int(batch_size), print_every=int(print_every),
                          beta_start=lam, beta_end=lam, alpha_start=lam, alpha_end=lam, **kwds)
        lstm.to("cpu")
        torch.save(lstm, ("char_model_%s.lstm" % str(lam)))
        meta_writer.add_scalar("HC-Ratio", r, (lam * 10000))
        ratios.append(r)
        lam += step_size
    torch.save(torch.tensor(r), "ratios.pt")


def linear_annealing(start, target, epochs):
    step_size = (target - start) / (epochs)
    current = start
    for i in range(epochs):
        current = current + step_size
        yield current


def elbow_annealing(start, target, epochs, elbow=.5):
    from math import floor
    step_size = (target - start) / (epochs * elbow)
    current = start
    for i in range(floor(epochs * elbow)):
        current = current + step_size
        yield current
    while True:
        yield target


def double_elbow_annealing(start, target, epochs, elbow_1=.25, elbow_2=.75):
    from math import floor
    rise_steps = epochs * (elbow_2 - elbow_1)
    step_size = (target - start) / rise_steps
    current = start
    for i in range(floor(epochs * elbow_1)):
        yield start
    for i in range(floor(rise_steps)):
        current = current + step_size
        yield current
    while True:
        yield target


def kl_divergence(mu_1, std_1, mu_2, std_2):
    std2_inv = std_2 ** (-1)
    mu_diff = mu_2 - mu_1
    dim = len(mu_2)
    kld = (1 / 2) * (std_2.log().sum() - std_1.log().sum(dim=1) - dim + (std_1 * std2_inv).sum(dim=1) +
                     (std2_inv * (mu_diff ** 2)).sum(dim=1))
    return kld


def split_list(data, length):
    new_list = []
    split_num = -(-len(data) // length) - 1
    for i in range(split_num):
        new_list.append(data[i * length:(i + 1) * length])
    new_list.append(data[split_num * length:])

    return new_list


if __name__ == '__main__':
    train_unimorph_lm(*sys.argv[1:])
    #train_many_both(*sys.argv[1:])
