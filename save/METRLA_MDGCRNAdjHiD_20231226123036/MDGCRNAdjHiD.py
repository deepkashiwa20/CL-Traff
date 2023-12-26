import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
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
        x_g = torch.cat(x_g, dim=-1) # B, N, 2 * cheb_k * dim_in
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


class MDGCRNAdjHiD(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, rnn_layers=1, cheb_k=3,
                 ycov_dim=1, mem_num=20, mem_dim=64, embed_dim=10, adj_mx=None, cl_decay_steps=2000, 
                 use_curriculum_learning=True, contra_loss='triplet', diff_max=3.74, diff_min=0, schema=1, device="cpu"):
        super(MDGCRNAdjHiD, self).__init__()
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
        self.contra_loss = contra_loss
        self.device = device
        self.diff_min = diff_min
        self.diff_max = diff_max
        self.schema = schema
        
        # memory
        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()
        
        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.rnn_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
        # graph
        self.adj_mx = adj_mx
        self.hypernet = nn.Sequential(nn.Linear(self.decoder_dim, self.embed_dim, bias=True))
        
        # latent distance
        if self.schema == 1:
            self.hypernet_lat = nn.Sequential(nn.Linear(self.mem_dim, 1, bias=True))  # for add / subtract
            # self.hypernet_lat = nn.Sequential(nn.Linear(2*self.mem_dim, 1, bias=True))  # for concat
        if self.schema == 3:
            self.hypernet_lat = nn.Sequential(nn.Linear(self.mem_dim, self.mem_dim, bias=True))  # mlp projection
        self.act_dict = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(), 'sigmoid': nn.Sigmoid()}
        self.act_fn = 'sigmoid'  # 'relu' 'lrelu' 'sigmoid'
        
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)     # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)    # project to query
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])     # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)         # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])     # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]] # B, N, d
        if self.contra_loss in ['infonce']:  # InfoNCE loss
            neg = self.memory['Memory'].repeat(query.shape[0], self.num_nodes, 1, 1)  # (B, N, M, d)
            mask_index = ind[:, :, [0]]  # B, N, 1
            mask = torch.zeros_like(att_score, dtype=torch.bool).to(att_score.device)  # B, N, M
            mask = mask.scatter(-1, mask_index, True)  
        elif self.contra_loss in ['triplet']:  # Triplet loss
            neg = self.memory['Memory'][ind[:, :, 1]] # B, N, d
            mask = None
        else:
            pass
        return value, query, pos, neg, mask
            
    def calculate_cosine(self, pos, pos_his):
        if self.schema == 3:
            pos, pos_his = self.hypernet_lat(pos), self.hypernet_lat(pos_his)  # B, N, d
        score = F.cosine_similarity(pos, pos_his, dim=-1)  # B, N
        
        return (1 - score) / 2  # normalized [0, 1]
    
    def forward(self, x, x_cov, x_his, y_cov, labels=None, batches_seen=None):
        supports_en = self.adj_mx
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)    
        h_att, query, pos, neg, mask = self.query_memory(h_t)    
        
        # TODO: for x_his
        h_his_en, state_his_en = self.encoder(x_his, init_state, supports_en) # B, T, N, hidden
        h_his_t = h_his_en[:, -1, :, :] # B, N, hidden (last state)      
        h_his_att, query_his, pos_his, neg_his, mask_his = self.query_memory(h_his_t)
        
        # TODO: detection loss
        # normalization [0, 1]
        real_dis = (torch.clamp(torch.abs(x-x_his)[:, -1, :, :].squeeze(-1), min=self.diff_min, max=self.diff_max) - self.diff_min) / (self.diff_max - self.diff_min) 
        if self.schema == 1:
            latent_dis = self.hypernet_lat(pos - pos_his).squeeze(-1)  # for add / subtract
            # latent_dis = self.hypernet_lat(torch.concat([pos, pos_his], dim=-1)).squeeze(-1)  # for concat
        else:
            latent_dis = self.calculate_cosine(pos, pos_his)
        latent_dis = self.act_dict.get(self.act_fn)(latent_dis)
        
        h_aug = torch.cat([h_t, h_att], dim=-1) # B, N, D
        
        node_embeddings = self.hypernet(h_aug) # B, N, e
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1) 
        supports_de = [support]
        
        ht_list = [h_aug]*self.rnn_layers
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
        
        return output, h_att, query, pos, neg, mask, real_dis, latent_dis

def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters.')
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
    model = MemDGCRN(num_nodes=args.num_variable, input_dim=args.channelin, output_dim=args.channelout, horizon=args.seq_len, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.his_len, args.num_variable, args.channelin), (args.seq_len, args.num_variable, args.channelout)], device=device)
    print_params(model)
    
if __name__ == '__main__':
    main()
