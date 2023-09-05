import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np

GRANULARITY = {
    'week': 2015,
    'day': 288,
    'hour': 12
}

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


class IndexDGCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, rnn_layers=1, cheb_k=3,
                 ycov_dim=1, embed_dim=10, cl_decay_steps=2000, use_curriculum_learning=True,
                 delta=10., sample=20, granu='week', temp=1., scaler=None):
        super(IndexDGCRN, self).__init__()
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
        self.delta = delta
        self.sample = sample
        self.granu = granu
        self.temp = temp
        self.scaler = scaler
        
        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.rnn_layers)
        
        # deocoder
        self.decoder_dim = self.rnn_units
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.rnn_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        
        # graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.hypernet = nn.Sequential(nn.Linear(self.rnn_units, self.embed_dim, bias=True))
        
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def get_pseudo_labels(self, x, x_his):
        x = self.scaler.inverse_transform(x)
        diff = (x - x_his).abs()
        pseudo_labels = (diff <= self.delta).squeeze(-1)  # (B, T, N) True means normal speed
        return pseudo_labels.sum(1) > 0  # regard T as whole
    
    def get_memory_labels(self, x, x_his):
        x = x.reshape(-1, self.sample, self.horizon, self.num_nodes, self.input_dim)  # (B*K, T, N, 1) -> (B, K, T, N, 1)
        diff = (x - x_his.unsqueeze(1)).abs()
        pseudo_labels = (diff <= self.delta).reshape(-1, self.horizon, self.num_nodes)  # (B*K, T, N) True means normal speed
        return pseudo_labels.sum(1) > 0  # regard T as whole
    
    def sample_history_memory(self, x_cov, memory):
        samples = []
        initial_times = x_cov[:, 0, 0, 0] * GRANULARITY[self.granu]  # (B, )
        for t in initial_times:
            sample_num = len(memory[int(t)])  #* t=873长度为0?!
            while sample_num == 0:
                t = t - 1
                sample_num = len(memory[int(t)])
            if self.sample >= sample_num:
                idxs = np.random.randint(0, sample_num, size=self.sample - sample_num)  # 有重复
                sample_t = memory[int(t)] + [memory[int(t)][id] for id in idxs]
            else:
                idxs = np.random.choice(sample_num, size=self.sample, replace=False)  # 无重复
                sample_t = torch.from_numpy(np.array([memory[int(t)][id] for id in idxs])).to(x_cov.device)
            samples.append(sample_t)  # (self.sample, T, N, 1)
        return torch.stack(samples, dim=0).reshape(-1, self.horizon, self.num_nodes, self.input_dim)  # (B * self.sample, T, N, 1)
    
    def filter_negative(self, input_, thres):
        times = input_[:, 0, 0, 0]
        m, m_inc = [], []
        cnt = 0
        # c = thres / 288
        c = thres / 2016
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
            
            res = torch.logical_or(st, gt).view(1, -1)  # exclude itself
            res_inc = res.clone()
            res_inc[0, cnt] = True  # include itself
            cnt += 1
            m.append(res)
            m_inc.append(res_inc)
        m = torch.cat(m)
        m_inc = torch.cat(m_inc)
        return m, m_inc
    
    def supervised_contrastive_loss(self, inputs, rep, labels, support):
        """
            inputs: input (bs, T, node, in_dim) in_dim=1, i.e., time slot
            rep: original representation, (bs, node, dim)
            labels: normal or abnormal flag, (bs, node), True: normal
            support: adaptive graph
            return: contra_loss, i.e., supervised contrastive loss
        """
        
        #* temporal contrast
        temporal_labels = labels.transpose(0, 1).unsqueeze(-1) ^ labels.transpose(0, 1).unsqueeze(1)  # (node, bs, bs)  # False: same label
        tempo_rep = rep.transpose(0, 1) # (node, bs, dim)
        temporal_matrix = torch.exp(torch.cosine_similarity(tempo_rep.unsqueeze(1), tempo_rep.unsqueeze(2), dim=-1) / self.temp)  # (node, bs, bs)

        # temporal negative filter
        if self.fn_t:  # easy negatives from temporal perspective, i.e., distant time
            temporal_mask_exc, temporal_mask_inc = self.filter_negative(inputs, self.fn_t)  # mask_exc (mask_inc) means that exclude (include) current time 
        #* same label as current node
        temporal_pos1 = temporal_matrix * ~temporal_labels * ~temporal_mask_inc  # regard those nodes with same label and close to current time as positives
        temporal_neg1 = temporal_matrix * ~temporal_labels * temporal_mask_exc # regard those nodes with same label and distant to current time as negatives
        #* different label from current node
        # regard those nodes with different label and close to current time as hard negatives, ignore them due to the difficulty!!!
        # then regard those nodes with different label and distant to current time as negatives
        temporal_neg2 = temporal_matrix * temporal_labels * temporal_mask_exc
        temporal_pos = torch.sum(temporal_pos1, dim=-1)  # (bs, node)
        temporal_neg = torch.sum(temporal_neg1, dim=-1) + torch.sum(temporal_neg2, dim=-1)

        #* spatial contrast
        spatial_labels = labels.unsqueeze(-1) ^ labels.unsqueeze(1)  # (bs, node, node)  # False: same label 
        spatial_matrix = torch.exp(torch.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(2), dim=-1) / self.temp)  # (bs, node, node)
        
        # first-order neighbor filter
        _, indices = torch.topk(support, k=self.top_k+1, dim=-1)  # (node, k+1)  # TODO cannot guarantee that the first index is in diagonal ?!
        spatial_mask = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool).to(inputs.device)
        spatial_mask[torch.arange(spatial_mask.size(0)).unsqueeze(1), indices[:, 1:]] = True  # exclude itself
        #* same label as current node
        spatial_pos1 = spatial_matrix * ~spatial_labels * spatial_mask  # regard those nodes with same label and close to current node as positives
        spatial_neg1 = spatial_matrix * ~spatial_labels * ~spatial_mask  # regard those nodes with same label and distant to current node as negatives
        #* different label from current node
        # regard those nodes with different label and close to current node as hard negatives, ignore them due to the difficulty!!!
        # then regard those nodes with different label and distant to current node as negatives
        spatial_neg2 = spatial_matrix * spatial_labels * ~spatial_mask
        spatial_pos = torch.sum(spatial_pos1, dim=-1)  # (bs, node)
        spatial_neg = torch.sum(spatial_neg1, dim=-1) + torch.sum(spatial_neg2, dim=-1)

        ratio = (temporal_pos.transpose(0,1) + spatial_pos + 1e-12) / (temporal_pos.transpose(0,1) + spatial_pos + temporal_neg.transpose(0,1) + spatial_neg)
        # ratio = (temporal_pos.transpose(0,1) + 1e-12) / (temporal_pos.transpose(0,1) + temporal_neg.transpose(0,1))  # only temporal
        # ratio = (spatial_pos + 1e-12) / (spatial_pos + spatial_neg)  # only spatial
        contra_loss = torch.mean(-torch.log(ratio))
        return contra_loss
         
    def memory_contrastive_loss(self, x_cov, h_t, h_t_memory, labels, memory_labels, support):
        h_t_memory = h_t_memory.reshape(-1, self.sample, self.num_nodes, self.rnn_units).transpose(1, 2)  # (B, K, N, D)->(B, N, K, D)
        memory_labels = memory_labels.reshape(-1, self.sample, self.num_nodes)  # (B, K, N)
        flags = labels.unsqueeze(1) ^ memory_labels  # (B, K, N)  False: same label
        sim_matrix = torch.exp(F.cosine_similarity(h_t.unsqueeze(2), h_t_memory, dim=-1).transpose(1, 2) / self.temp)  # (B, K, N)
        pos_score = (sim_matrix * ~flags).sum(1) 
        neg_score = (sim_matrix * flags).sum(1) 
        ratio = (pos_score + 1e-12) / (pos_score + neg_score)
        contra_loss = torch.mean(-torch.log(ratio))
        return contra_loss
    
    def forward(self, x, y_cov, labels=None, x_cov=None, x_his=None, y_his=None, batches_seen=None, memory=None):
        support = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        supports_en = [support]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports_en) # B, T, N, hidden
        h_t = h_en[:, -1, :, :] # B, N, hidden (last state)        
        
        #* supervised contrastive learning 
        contra_loss = None
        if labels is not None and x_his is not None:
            # sample history memory
            history_sample = self.sample_history_memory(x_cov, memory)  # (B*K, T, N, 1)
            pseudo_labels = self.get_pseudo_labels(x, x_his)  # (B, N)
            memory_labels = self.get_memory_labels(history_sample, x_his)  # (B*K, N)
            init_state = self.encoder.init_hidden(history_sample.shape[0])
            h_en_memory, state_en_memory = self.encoder(history_sample, init_state, supports_en) # B, T, N, hidden
            h_t_memory = h_en_memory[:, -1, :, :] # B*K, N, hidden (last state) 
            contra_loss = self.memory_contrastive_loss(x_cov, h_t, h_t_memory, pseudo_labels, memory_labels, support)
        
        node_embeddings = self.hypernet(h_t) # B, N, d
        support = F.softmax(F.relu(torch.einsum('bnc,bmc->bnm', node_embeddings, node_embeddings)), dim=-1) 
        supports_de = [support]
        
        ht_list = [h_t]*self.rnn_layers
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
        
        return output, contra_loss

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
