import torch
import torch.nn as nn
import numpy as np


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, num_support):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(
            torch.FloatTensor(num_support * cheb_k * dim_in, dim_out)
        )  # num_support*cheb_k*dim_in is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        for support in supports:
            if len(support.shape) == 2:
                support_ks = [torch.eye(support.shape[0]).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(
                        torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]
                    )
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                support_ks = [
                    torch.eye(support.shape[1])
                    .repeat(support.shape[0], 1, 1)
                    .to(support.device),
                    support,
                ]
                for k in range(2, self.cheb_k):
                    support_ks.append(
                        torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]
                    )
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = (
            torch.einsum("bni,io->bno", x_g, self.weights) + self.bias
        )  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_support):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, num_support)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Encoder, self).__init__()
        assert rnn_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support)
        )
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
            )

    def forward(self, x, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (rnn_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.rnn_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (rnn_layers, B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers, num_support):
        super(ADCRNN_Decoder, self).__init__()
        assert rnn_layers >= 1, "At least one DCRNN layer in the Decoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(
            AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support)
        )
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
            )

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (rnn_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.rnn_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MDGCRN_woSSDL(nn.Module):
    def __init__(
        self,
        num_nodes=207,
        input_dim=1,
        output_dim=1,
        horizon=12,
        rnn_units=128,
        rnn_layers=1,
        cheb_k=3,
        ycov_dim=1,
        embed_dim=10,
        adj_mx=None,
        tf_decay_steps=2000,
        use_teacher_forcing=True,
        use_STE=False,
        device="cpu",
    ):
        super(MDGCRN_woSSDL, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.embed_dim = embed_dim
        self.tf_decay_steps = tf_decay_steps
        self.use_teacher_forcing = use_teacher_forcing
        self.device = device
        self.use_STE = use_STE
        self.TDAY = 288

        # projection & spatio-temporal embedding
        if self.use_STE:
            # projection
            self.input_proj = nn.Linear(self.input_dim, self.rnn_units)
            self.node_embedding = nn.Parameter(
                torch.empty(self.num_nodes, self.embed_dim)
            )
            self.time_embedding = nn.Parameter(torch.empty(self.TDAY, self.embed_dim))
            nn.init.xavier_uniform_(self.node_embedding)
            nn.init.xavier_uniform_(self.time_embedding)

        # encoder
        self.adj_mx = adj_mx
        if self.use_STE:
            self.encoder = ADCRNN_Encoder(
                self.num_nodes,
                self.rnn_units + self.embed_dim * 2,
                self.rnn_units,
                self.cheb_k,
                self.rnn_layers,
                len(self.adj_mx),
            )
        else:
            self.encoder = ADCRNN_Encoder(
                self.num_nodes,
                self.input_dim,
                self.rnn_units,
                self.cheb_k,
                self.rnn_layers,
                len(self.adj_mx),
            )

        # deocoder
        self.decoder_dim = self.rnn_units
        if self.use_STE:
            self.decoder = ADCRNN_Decoder(
                self.num_nodes,
                self.rnn_units + self.embed_dim * 2,
                self.decoder_dim,
                self.cheb_k,
                self.rnn_layers,
                1,
            )
        else:
            self.decoder = ADCRNN_Decoder(
                self.num_nodes,
                self.output_dim + self.ycov_dim,
                self.decoder_dim,
                self.cheb_k,
                self.rnn_layers,
                1,
            )

        # output
        self.proj = nn.Linear(self.decoder_dim, self.output_dim)

        # graph
        # self.hypernet = nn.Linear(self.rnn_units*2, self.embed_dim)
        self.hypernet = nn.Linear(self.rnn_units, self.embed_dim)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
            self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, x, x_cov, x_his, y_cov, labels=None, batches_seen=None):
        if self.use_STE:
            x = self.input_proj(x)  # [B,T,N,1]->[B,T,N,D]
            # x_his = self.input_proj(x_his)  # [B,T,N,1]->[B,T,N,D]
            node_emb = self.node_embedding.expand(
                x.shape[0], self.horizon, *self.node_embedding.shape
            )  # [B,T,N,d]
            time_emb = self.time_embedding[
                (x_cov.squeeze() * self.TDAY).long()
            ]  # [B, T, N, d]
            x = torch.cat([x, node_emb, time_emb], dim=-1)  # [B, T, N, D+2d]
            # x_his = torch.cat([x_his, node_emb, time_emb], dim=-1)  # [B, T, N, D+2d]

        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        # for x_his
        # h_his_en, state_his_en = self.encoder(
        #     x_his, init_state, supports_en
        # )  # B, T, N, hidden
        # h_his_t = h_his_en[:, -1, :, :]  # B, N, hidden (last state)

        h_de = h_t
        # h_aug = torch.cat([h_t, h_his_t], dim=-1)  # B, N, D
        h_aug = h_t

        node_embeddings = self.hypernet(h_aug)  # B, N, e
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", node_embeddings, node_embeddings)),
            dim=-1,
        )
        supports_de = [support]

        ht_list = [h_de] * self.rnn_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)

        out = []
        for t in range(self.horizon):
            if self.use_STE:
                go = self.input_proj(go)  # equal to torch.zeros(B,N,D)
                node_emb = self.node_embedding.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B,N,d]
                time_emb = self.time_embedding[(y_cov[:, t, ...].squeeze() * self.TDAY).type(torch.LongTensor)]  # [B, N, d]
                go = torch.cat([go, node_emb, time_emb], dim=-1)  # [B, N, D+2d]
                h_de, ht_list = self.decoder(go, ht_list, supports_de)
            else:
                h_de, ht_list = self.decoder(
                    torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports_de
                )
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_teacher_forcing:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)

        return output, None, None


if __name__ == "__main__":
    from torchinfo import summary
    from utils import load_adj

    adj_mx = load_adj("../METRLA/adj_mx.pkl", "symadj")
    adj_mx = [torch.FloatTensor(i) for i in adj_mx]
    model = MDGCRN_woSSDL(adj_mx=adj_mx, use_STE=True)
    summary(
        model,
        [[8, 12, 207, 1], [8, 12, 207, 1], [8, 12, 207, 1], [8, 12, 207, 1]],
        device="cpu",
    )
