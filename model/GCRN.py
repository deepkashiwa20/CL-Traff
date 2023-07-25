import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import load_pickle, sym_adj

class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, support):
        node_num = support.shape[0]
        support_set = [torch.eye(node_num).to(support.device), support]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * support, support_set[-1]) - support_set[-2]) 
        
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1) # B, N, cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv
    
class GCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(GCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, support):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, support))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    
class GCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(GCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one GCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.gcrnn_cells = nn.ModuleList()
        self.gcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.gcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, support):
        #shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.gcrnn_cells[i](current_inputs[:, t, :, :], state, support)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        #current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        #output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        #last_state: (B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden
    
    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.gcrnn_cells[i].init_hidden_state(batch_size))
        return init_states

class GCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(GCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one GCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.gcrnn_cells = nn.ModuleList()
        self.gcrnn_cells.append(GCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.gcrnn_cells.append(GCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, support):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.gcrnn_cells[i](current_inputs, init_state[i], support)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class GCRN(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, horizon, rnn_units, num_layers=1, embed_dim=8, cheb_k=3, ycov_dim=1, cl_decay_steps=2000, 
                 use_curriculum_learning=True, fn_t=12, temp=0.1, top_k=10, input_masking_ratio=0.01, fusion_num=1, schema=1,  
                 contrastive_denominator=False, device="cpu"):
        super(GCRN, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.cheb_k = cheb_k
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.ycov_dim = ycov_dim
        self.decoder_dim = self.rnn_units
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        # TODO: support contrastive learning
        self.fn_t = fn_t
        self.temp = temp
        self.top_k = top_k
        self.device = device
        self.im_t = input_masking_ratio
        self.fusion_num = fusion_num
        self.schema = schema
        self.contrastive_denominator = contrastive_denominator
        
        # graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        
        # encoder
        self.encoder = GCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        
        # TODO: support different augmentation schemas
        #* schema 1: two different encoders
        if self.schema in [1, 3]:
            self.encoder_aug = GCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        
        #* schema 2: only one encoder but with two different MLP encoders
        if self.schema == 2: 
            self.fc_aug1_mean = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                )
            self.fc_aug1_var = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                )
            self.fc_aug2_mean = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                )
            self.fc_aug2_var = nn.Sequential(
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                )
        #* schema 3: two different encoders but with fused representations for decoder (fusion is worse)
        if self.schema == 3:
            if self.fusion_num == 1:
                self.fusion_layer = nn.Sequential(
                    nn.Linear(self.decoder_dim * 2, self.decoder_dim),
                )
            else:
                self.fusion_layer = nn.Sequential(
                    nn.Linear(self.decoder_dim * 2, self.decoder_dim),
                    nn.ReLU(),
                    nn.Linear(self.decoder_dim, self.decoder_dim),
                )
        # deocoder
        self.decoder = GCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k, self.num_layers)
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))
    
    def filter_negative(self, input_, thres):
        times = input_[:, 0, 0, 0]
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
            # res[0, cnt] = True
            # cnt += 1
            m.append(res)
        m = torch.cat(m)
        return m
    
    def get_unsupervised_loss(self, inputs, rep, rep_aug, support):
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
        tempo_matrix = torch.exp(tempo_matrix / self.temp)

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
            _, indices = torch.topk(support, k=self.top_k+1, dim=-1)  # (node, k)
            adj = torch.ones((self.num_nodes, self.num_nodes), dtype=torch.bool).to(self.device)
            adj[torch.arange(adj.size(0)).unsqueeze(1), indices] = False
            adj = adj + diag
            spatial_matrix = spatial_matrix * adj
        spatial_neg = torch.sum(spatial_matrix, dim=2) # (bs, node)

        if not self.contrastive_denominator:
            ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0,1) - pos_sum)
        else:
         ratio = pos_sum / (spatial_neg + tempo_neg.transpose(0,1))
        # ratio = pos_sum / tempo_neg.transpose(0,1)  #* no spatial_neg is not better than with spatial_neg
        u_loss = torch.mean(-torch.log(ratio))
        return u_loss    
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
    
    def forward(self, x, x_cov, y_cov, labels=None, batches_seen=None):
        support = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, support) # B, T, N, hidden      
        h_t = h_en[:, -1, :, :]   # B, N, hidden (last state)        
        ht_list = [h_t]*self.num_layers
        
        # TODO: support different augmentation schemas
        #* two encoders
        if labels is not None and self.schema in [1, 3]:
            init_state_aug = self.encoder_aug.init_hidden(x.shape[0])
            h_en_aug, state_en_aug = self.encoder_aug(x, init_state_aug, support) # B, T, N, hidden      
            h_t_aug = h_en_aug[:, -1, :, :]   # B, N, hidden (last state)
        
        #* one encoder with extra two mlp encoders
        if labels is not None and self.schema == 2:
            h_t_aug = self.sampling(self.fc_aug1_mean(h_t), self.fc_aug1_var(h_t))
            h_t = self.sampling(self.fc_aug2_mean(h_t), self.fc_aug2_var(h_t))
        
        #* fusion for decoder
        if labels is not None and self.schema == 3:  
            ht_list_aug = [h_t_aug]*self.num_layers
            ht_list = [self.fusion_layer(torch.cat(ht_list + ht_list_aug, dim=-1))]*self.num_layers
        
        #* one encoder with input data augmentation
        if labels is not None and self.schema == 4:
            x_ = x.detach().clone()
            bs, frame, num_node, _ = x.size()
            rand = torch.rand(bs, frame, num_node).to(self.device)
            x_[:,:,:,0] = x_[:,:,:,0] * (rand >= self.im_t)
            init_state_aug = self.encoder.init_hidden(x_.shape[0])
            h_en_aug, state_en_aug = self.encoder(x_, init_state_aug, support) # B, T, N, hidden      
            h_t_aug = h_en_aug[:, -1, :, :]   # B, N, hidden (last state) 
            
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, support)
            go = self.proj(h_de)
            out.append(go)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        
        # TODO self-supervised contrastive learning
        if labels is not None and self.schema in [1, 2, 3, 4]:
            u_loss = self.get_unsupervised_loss(x_cov, h_t, h_t_aug, support)
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
    print(f'In total: {param_count} trainable parameters.')
    return

def main():
    import sys
    import argparse
    from torchsummary import summary
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0, help="which GPU to use")
    parser.add_argument('--num_nodes', type=int, default=207, help='number of variables (e.g., 207 in METR-LA, 325 in PEMS-BAY)')
    parser.add_argument('--seq_len', type=int, default=12, help='sequence length of historical observation')
    parser.add_argument('--horizon', type=int, default=12, help='sequence length of prediction')
    parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
    parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
    parser.add_argument('--rnn_units', type=int, default=64, help='number of hidden units')
    args = parser.parse_args()
    device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
    model = GCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, 
                 horizon=args.horizon, rnn_units=args.rnn_units).to(device)
    summary(model, [(args.seq_len, args.num_nodes, args.input_dim), (args.seq_len, args.num_nodes, args.input_dim), (args.horizon, args.num_nodes, args.output_dim)], device=device)
    print_params(model)
        
if __name__ == '__main__':
    main()