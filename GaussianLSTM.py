import torch
import torch.nn as nn
from enum import IntEnum
import sys
import math
import itertools

import numpy as np
import random

INF = float('inf')
LOG2 = math.log(2)

PADDING = '<!PAD!>'
EOS = '<!EOS!>'

PADDING_IDX = 0
EOS_IDX = 1

flat = itertools.chain.from_iterable


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class GaussianLSTMCore(nn.Module):
    def __init__(self, input_sz: int, hidden_sz: int, noise: str) \
            :
        super(GaussianLSTMCore, self).__init__()
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_ih = nn.Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.noise = noise

        if noise == ("c" or "both"):
            self.c_reparameterize = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 2))
        if noise == ("h" or "both"):
            self.h_reparameterize = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz * 2))

        self.bias = nn.Parameter(torch.Tensor(hidden_sz * 4))
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)
        if hasattr(self,"h_reparameterize"):
            scaling = torch.cat([torch.ones(self.hidden_size, self.hidden_size), torch.ones(self.hidden_size, self.hidden_size) * .000000001], dim=1)
            self.h_reparameterize = torch.nn.Parameter(self.h_reparameterize * scaling)

        if hasattr(self,"c_reparameterize"):
            scaling = torch.cat([torch.ones(self.hidden_size, self.hidden_size), torch.ones(self.hidden_size, self.hidden_size) * .000000001], dim=1)
            self.c_reparameterize = torch.nn.Parameter(self.c_reparameterize * scaling)

    def forward(self, x: torch.Tensor,
                init_states=None):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        c_stats = []
        h_stats = []
        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        hidden_seq = []
        c_seq = []

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[:, t, :]
            # batch the computations into a single matrix multiplication
            gates = x_t @ self.weight_ih + h_t @ self.weight_hh + self.bias
            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]),  # input
                torch.sigmoid(gates[:, HS:HS * 2]),  # forget
                torch.tanh(gates[:, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, HS * 3:]),  # output
            )
            pre_c_t = f_t * c_t + i_t * g_t
            pre_h_t = o_t * torch.tanh(c_t)

            # Produce parameters for Gaussian distribution of hidden_size dimensions
            # sample from Gaussian distribution
            # otherwise, pass pre_c_t/pre_h_t on as c_t/h_t itself
            if self.noise == ("h" or "both"):
                h_params = pre_h_t @ self.h_reparameterize
                h_mu, h_std, h_t = self.sample(h_params)
                h_stats.append(torch.cat([h_mu, h_std], dim=Dim.seq).unsqueeze(Dim.batch))
            else:
                h_t = pre_h_t
            if self.noise == ("c" or "both"):
                c_params = pre_c_t @ self.c_reparameterize
                c_mu, c_std, c_t = self.sample(c_params)
                c_stats.append(torch.cat([c_mu, c_std], dim=Dim.seq).unsqueeze(Dim.batch))
            else:
                c_t = pre_c_t

            # append returnable sequences
            c_seq.append(c_t.unsqueeze(Dim.batch))
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

            # c_stats.append(torch.cat([c_mu, c_std],dim=Dim.seq).unsqueeze(Dim.batch))
            # h_stats.append(torch.cat([h_mu, h_std],dim=Dim.seq).unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        c_seq = torch.cat(c_seq, dim=Dim.batch)

        # Improvable later, but this section transforms and splits the stats tensors if relevant
        # Tensor shape B x T x H
        if self.noise == ("h" or "both"):
            h_stats = torch.cat(h_stats, dim=Dim.batch)
            h_stats = h_stats.transpose(Dim.batch, Dim.seq).contiguous()
            h_stds = h_stats[:, :, self.hidden_size:].contiguous()
            h_mus = h_stats[:, :, :self.hidden_size].contiguous()
        else:
            h_mus = None
            h_stds = None
        if self.noise == ("c" or "both"):
            c_stats = torch.cat(c_stats, dim=Dim.batch)
            c_stats = c_stats.transpose(Dim.batch, Dim.seq).contiguous()
            c_stds = c_stats[:, :, self.hidden_size:].contiguous()
            c_mus = c_stats[:, :, :self.hidden_size].contiguous()
        else:
            c_mus = None
            c_stds = None

        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        c_seq = c_seq.transpose(Dim.batch, Dim.seq).contiguous()
        # h_stats = h_stats.transpose(Dim.batch, Dim.seq).contiguous()
        # c_stats = c_stats.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, c_seq, (h_mus, h_stds), (c_mus, c_stds), (h_t, c_t)

    def sample(self, stats):
        mu = stats[:, :self.hidden_size]
        std = nn.functional.softplus(stats[:, self.hidden_size:], beta=1)
        eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
        # output noisy sampling of size B x T x H
        return mu, std, mu + eps * std


class LSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_layers, hidden_size, noise, padding_idx=PADDING_IDX,
                 eos_idx=EOS_IDX):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.eos_idx = eos_idx
        self.noise = noise

        self.__build_model()

    def __build_model(self):
        self.word_embedding = torch.nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_dim,
            padding_idx=self.padding_idx
        )


        # For the creation of multi-layer LSTMs
        g = [GaussianLSTMCore(self.embedding_dim, self.hidden_size, self.noise)]
        g.extend([GaussianLSTMCore(self.hidden_size, self.hidden_size, self.noise) for _ in range(1, self.num_layers)])
        self.gauss_lstm = nn.ModuleList(g)

        # for i in range(self.num_layers):
        #     g = GaussianLSTMCore(self.embedding_dim,self.hidden_size,"h")
        #     print(type(g))
        #     self.gauss_lstm.append(g)

        # self.gauss_lstm = GaussianLSTMCore(
        #     input_sz=self.embedding_dim,
        #     hidden_sz=self.hidden_size,
        #     noise="h"
        #     #,
        #     #batch_first=True
        # )

        self.initial_state = (
            torch.autograd.Variable(torch.randn(self.num_layers, self.hidden_size), requires_grad=True),
            torch.autograd.Variable(torch.randn(self.num_layers, self.hidden_size), requires_grad=True)
        )

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
        for i, layer in enumerate(self.gauss_lstm):
            seq, c_seq, (h_mu, h_std), (c_mu, c_std), (_, __) = layer(seq, (init_h[i], init_c[i]))
            c_mus.append(c_mu)
            c_stds.append(c_std)
            h_mus.append(h_mu)
            h_stds.append(h_std)

        # If noise flag "h"/"c", transforms them, otherwise don't touch because it's a NoneType

        if self.noise == ("c" or "both"):
            c_mus = torch.cat(c_mus, Dim.feature)
            c_stds = torch.cat(c_stds, Dim.feature)

        if self.noise == ("h" or "both"):
            h_mus = torch.cat(h_mus, Dim.feature)
            h_stds = torch.cat(h_stds, Dim.feature)

        c_stats = (c_mus, c_stds)
        h_stats = (h_mus, h_stds)

        # Also need to include the initial state itself
        # And the last state is after consuming EOS, so it doesn't matter
        # So add the initial state in first, and remove the final state

        seq = torch.cat((init_h[-1].unsqueeze(-2), seq), dim=-2)[:, 0:-1, :]
        c_seq = torch.cat((init_h[-1].unsqueeze(-2), c_seq), dim=-2)[:, 0:-1, :]

        return seq, c_seq, h_stats, c_stats

    def decode(self, h):
        logits = self.decoder(h)
        # Remove probability mass from the pad token
        logits[:, :, self.padding_idx] = -INF
        return torch.log_softmax(logits, -1)

    def lm_loss(self, Y, Y_hat):
        # Y contains the target token indices. Shape B x T
        # Y_hat contains distributions. Shape B x (T+1) x V ??? (Doesn't appear to be true)
        Y_flat = Y.view(-1)  # flatten to B*T
        Y_hat_flat = Y_hat.view(-1, self.vocab_size)  # flatten to B*T x V
        mask = (Y_flat == self.padding_idx)
        num_tokens = torch.sum(~mask).item()
        Y_hat_correct = Y_hat_flat[range(Y_hat_flat.shape[0]), Y_flat]
        Y_hat_correct[mask] = 0
        ce_loss = -torch.sum(Y_hat_correct) / num_tokens
        return ce_loss

    def mi_loss(self, stats, Y):
        std = stats[1]  # stats[:,:,self.hidden_size:].contiguous()
        mu = stats[0]  #[:, :, :self.hidden_size].contiguous()
        Y_flat = Y.view(-1)
        mask = (Y_flat == self.padding_idx)

        std_flat = std.contiguous().view(-1, self.hidden_size * self.num_layers)[mask, :] # B*TxH, remove padded entries
        mu_flat = mu.contiguous().view(-1, self.hidden_size * self.num_layers)[mask, :] # B*TxH, ditto

        mi_loss = -(1 / 2) * (1 + 2 * std_flat.log() - mu_flat ** 2 - std_flat ** 2).mean()

        return mi_loss

    def train_lm(self, data, print_every=10, num_epochs=1000, batch_size=None, beta=.05, **kwds):
        if batch_size is None:
            batch_size = len(data)
        opt = torch.optim.Adam(lr = .01, params=self.parameters(), **kwds)
        for i in range(num_epochs):
            opt.zero_grad()
            batch = random.sample(data, batch_size)  # shape B x T
            padded_batch = pad_sequences(batch)
            hidden_seq, c_seq, h_stats, c_stats = self.encode(padded_batch)  # shape B x (T+1) x H
            y_hat = self.decode(hidden_seq)  # shape B x (T+1) x V ???
            #y = torch.roll(padded_batch, -1, -1)  # shape B x T
            hib_loss = self.mi_loss(h_stats, padded_batch)
            ce_loss = self.lm_loss(padded_batch, y_hat)
            loss = beta * hib_loss + ce_loss
            loss.backward()
            opt.step()
            if i % print_every == 0:
                print("epoch %d, loss = %s, mutual information = %s" % (
                i, str(loss.item() / LOG2), str(hib_loss.item() / LOG2)), file=sys.stderr)

    def distro_after(self, sequence):
        sequence = list(sequence) + [PADDING_IDX]
        padded = pad_sequences([sequence])
        hidden_seq, c_seq, h_stats, c_stats = self.encode(padded)
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
    lstm.train_lm(data, num_epochs = 2000, **kwds)
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
    return result, indexer.vocab


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


def train_unimorph_lm(lang, hidden_size=100, num_layers=2, batch_size = 50, num_epochs=20000, print_every=200,
                      num_samples=5, **kwds):
    data, vocab = list(format_sequences(read_unimorph("%s" % lang)))
    print("Loaded data for %s..." % lang, file=sys.stderr)
    vocab_size = len(vocab)
    print("Vocab size: %d" % vocab_size, file=sys.stderr)
    lstm = LSTM(vocab_size, 20, num_layers, hidden_size, noise = "h")
    print(lstm, file=sys.stderr)
    lstm.train_lm(data, num_epochs=num_epochs, batch_size=batch_size, print_every=print_every, beta=0, **kwds)
    print("Generating %d samples..." % num_samples, file=sys.stderr)
    for _ in range(num_samples):
        symbols = list(lstm.generate())[:-1]
        print("".join(map(vocab.__getitem__, symbols)), file=sys.stderr)
    return lstm, vocab


if __name__ == '__main__':
    train_unimorph_lm(*sys.argv[1:])
