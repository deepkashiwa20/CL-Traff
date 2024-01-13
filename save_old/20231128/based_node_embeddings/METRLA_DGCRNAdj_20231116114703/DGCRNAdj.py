import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k): # this can be extended to (self, dim_in, dim_out, cheb_k, num_support)
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out)) # num_support*cheb_k*dim_in is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        for support in supports:
            if len(support.shape) == 2:
                support_ks = [torch.eye(support.shape[0]).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
                for graph in support_ks:
                    x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
            else:
                support_ks = [torch.eye(support.shape[1]).repeat(support.shape[0], 1, 1).to(support.device), support]
                for k in range(2, self.cheb_k):
                    support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
                for graph in support_ks:
                    x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = AGCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        #shape of x: (B, T, N, D), shape of init_state: (rnn_layers, B, N, hidden_dim)
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
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        #output_hidden: the last state for each layer: (rnn_layers, B, N, hidden_dim)
        #return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.rnn_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, rnn_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert rnn_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.rnn_layers = rnn_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, rnn_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

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


class DGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, rnn_layers=1, cheb_k=3,
                 ycov_dim=1, embed_dim=10, adj_mx=None, cl_decay_steps=2000, use_curriculum_learning=True,
                 fn_t=12, temp=0.1, top_k=10, schema=1, contra_denominator=True, use_graph=False, device="cpu"):
        super(DGCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.rnn_layers = rnn_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.embed_dim = embed_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        # TODO: support contrastive learning
        self.fn_t = fn_t
        self.temp = temp
        self.top_k = top_k
        self.device = device
        self.schema = schema
        self.contra_denominator = contra_denominator
        self.use_graph = use_graph
        
        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.rnn_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
        # graph
        self.adj_mx = adj_mx
        self.hypernet = nn.Sequential(nn.Linear(self.rnn_units, self.embed_dim, bias=True))
        
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def filter_negative(self, input_, thres):
        times = input_[:, 0, 0, 0]
        m = []
        cnt = 0
        c = thres / 288
        # c = thres / 2016
        for t in times:
            if t < c:
                st = times < 0
                gt = torch.logical_and(times <= (1 + t - c), times >= (t + c))
            elif t > (1 - c):
                st = torch.logical_and(times <= (t - c), times >= (c + t - 1))
                gt = times > 1
            else:
                st = times <= (t - c)
                gt = times >= (t + c)
            
            res = torch.logical_or(st, gt).view(1, -1)
            # res[0, cnt] = True
            # cnt += 1
            m.append(res)
        m = torch.cat(m)
        return m
    
    def filter_negative_graph(self, input_, thres):
        times = input_[:, 0, 0, 1]
        m = []
        cnt = 0
        c = thres / 288
        for t in times:
            if t < c:
                st = times < 0
                gt = torch.logical_and(times <= (1 + t - c), times >= (t + c))
            elif t > (1 - c):
                st = torch.logical_and(times <= (t - c), times >= (c + t - 1))
                gt = times > 1
            else:
                st = times <= (t - c)
                gt = times >= (t + c)
            
            res = torch.logical_or(st, gt).view(1, -1)
            res[0, cnt] = True
            cnt += 1
            m.append(res)
        m = torch.cat(m)
        return m
    
    def get_unsupervised_loss(self, inputs, rep, rep_aug, supports_en):
        """
            inputs: input (bs, T, node, in_dim) in_dim=1, i.e., time slot
            rep: original representation, (bs, node, dim)
            rep_aug: its augmented representation, (bs, node, dim)
            return: u_loss, i.e., unsupervised contrastive loss
        """
        # temporal contrast
        tempo_rep = rep.transpose(0,1) # (node, bs, dim)
        tempo_rep_aug = rep_aug.transpose(0,1)
        tempo_norm = tempo_rep.norm(dim=2).unsqueeze(dim=2)
        tempo_norm_aug = tempo_rep_aug.norm(dim=2).unsqueeze(dim=2)
        tempo_matrix = torch.matmul(tempo_rep, tempo_rep_aug.transpose(1,2)) / torch.matmul(tempo_norm, tempo_norm_aug.transpose(1,2))
        tempo_matrix = torch.exp(tempo_matrix / self.temp)  # (node, bs, bs)

        # temporal negative filter
        if self.fn_t:
            m = self.filter_negative(inputs, self.fn_t)
            tempo_matrix = tempo_matrix * m
        tempo_neg = torch.sum(tempo_matrix, dim=2) # (node, bs)

        # spatial contrast
        spatial_norm = rep.norm(dim=2).unsqueeze(dim=2)
        spatial_norm_aug = rep_aug.norm(dim=2).unsqueeze(dim=2)
        spatial_matrix = torch.matmul(rep, rep_aug.transpose(1,2)) / torch.matmul(spatial_norm, spatial_norm_aug.transpose(1,2))
        spatial_matrix = torch.exp(spatial_matrix / self.temp)  # (bs, node, node)
        
        diag = torch.eye(self.num_nodes, dtype=torch.bool).to(self.device)
        pos_sum = torch.sum(spatial_matrix * diag, dim=2) # (bs, node)
        
        # spatial negative filter
        if self.fn_t:
            adj = (supports_en[0] == 0).to(self.device)  # True means distant nodes
            adj = adj + diag
            spatial_matrix = spatial_matrix * adj
        spatial_neg = torch.sum(spatial_matrix, dim=2) # (bs, node)

        if not self.contra_denominator:
            ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0,1) - pos_sum)
        else:
            ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0,1))
        # ratio = pos_sum / tempo_neg.transpose(0,1)
        u_loss = torch.mean(-torch.log(ratio))
        return u_loss
    
    def get_unsupervised_loss_graph(self, inputs, rep, rep_aug, supports_en):
        """
            inputs: input (bs, T, node, in_dim) in_dim=1, i.e., time slot
            rep: original representation, (bs, dim)
            rep_aug: its augmented representation, (bs, dim)
            return: u_loss, i.e., unsupervised contrastive loss
        """
        # temporal contrast
        norm = rep.norm(dim=1)
        norm_aug = rep_aug.norm(dim=1)
        sim_matrix = torch.mm(rep, rep_aug.transpose(0,1)) / torch.mm(norm.view(-1, 1), norm_aug.view(1,-1))
        sim_matrix = torch.exp(sim_matrix / self.temp)  # (bs, bs)

        diag = torch.eye(inputs.shape[0], dtype=torch.bool).to(self.device)
        pos_sum = torch.sum(sim_matrix * diag, dim=1) # (bs, )
        
        # temporal negative filter
        if self.fn_t:
            m = self.filter_negative_graph(inputs, self.fn_t)  # regard each node as its negative
            sim_matrix = sim_matrix * m
        
        if not self.contra_denominator:
            ratio = pos_sum / (sim_matrix.sum(dim=1) - pos_sum)
        else:
            ratio = pos_sum / sim_matrix.sum(dim=1)
            
        u_loss = torch.mean(-torch.log(ratio))
        return u_loss
            
    def forward(self, x, x_cov, y_cov, labels=None, batches_seen=None):
        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)    
        ht_list = [h_t]*self.rnn_layers    
        
        node_embeddings = self.hypernet(h_t) # B, N, d
        h_t = node_embeddings.sum(dim=1) if self.use_graph else node_embeddings
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1) 
        supports_de = [support]
        
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports_de)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        
        # TODO self-supervised contrastive learning
        if labels is not None and self.schema in [1]:
            if not self.use_graph:
                u_loss = self.get_unsupervised_loss(x_cov, h_t, h_t, supports_en)
            else:
                u_loss = self.get_unsupervised_loss_graph(x_cov, h_t, h_t, supports_en)
            return output, u_loss
        return output, None

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters. \n')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="which GPU to use")
    parser.add_argument('--num_variable', type=int, default=207, help='number of variables (e.g., 207 in METR-LA, 325 in PEMS-BAY)')
    parser.add_argument('--his_len', type=int, default=12, help='sequence length of historical observation')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length of prediction')
    parser.add_argument('--channelin', type=int, default=1, help='number of input channel')
    parser.add_argument('--channelout', type=int, default=1, help='number of output channel')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of hidden units')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = DGCRN(num_nodes=args.num_variable, input_dim=args.channelin, output_dim=args.channelout, horizon=args.seq_len, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.his_len, args.num_variable, args.channelin), (args.seq_len, args.num_variable, args.channelout)], device=device)
    print_params(model)
    
if __name__ == '__main__':
    main()
