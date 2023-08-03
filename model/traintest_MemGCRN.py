import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import argparse
import logging
from utils import StandardScaler, DataLoader, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss
from MemGCRN import MemGCRN

class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.5, contra_denominator=True):
        super().__init__()
        self.temp = temp
        self.contra_denominator = contra_denominator
    
    def forward(self, pos_rep, neg_rep, mask):
        """
            pos: (B, N, D)
            neg: (B, N, M, D)
            mask: (B, N, M), where 0 means positive indices, 1 means negative indices
            return: InfoNCE loss, i.e., unsupervised contrastive loss
        """
        sim_score = torch.exp(F.cosine_similarity(pos_rep.unsqueeze(2), neg_rep, dim=-1) / self.temp)  # B, N, M
        pos_score = torch.sum(~mask * sim_score, dim=-1)  # B, N
        neg_score = torch.sum(mask * sim_score, dim=-1)  # B, N
        if self.contra_denominator:
            ratio = pos_score / (pos_score + neg_score)  # B, N
        else:
            ratio = pos_score / neg_score 
        return torch.mean(-torch.log(ratio))
        
    
def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():  
    model = MemGCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon, 
                 rnn_units=args.rnn_units, rnn_layers=args.rnn_layers, embed_dim=args.embed_dim, cheb_k=args.max_diffusion_step, 
                 mem_num=args.mem_num, mem_dim=args.mem_dim, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning,
                 delta=args.delta, method=args.method, contra_denominator=args.contra_denominator, scaler=None).to(device)
    return model

def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., 0:1]  # speed
    x1 = x[..., 1:2]  # weekdaytime 
    x2 = x[..., 2:3]  # history_speed
    y0 = y[..., 0:1]
    y1 = y[..., 1:2]
    y2 = y[..., 2:3]
    x0 = torch.from_numpy(x0).float()
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    y0 = torch.from_numpy(y0).float()
    y1 = torch.from_numpy(y1).float()
    y2 = torch.from_numpy(y2).float()
    return x0.to(device), x1.to(device), x2.to(device), y0.to(device), y1.to(device), y2.to(device) # x, x_cov, x_his, y, y_cov, y_his
    
def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter =  data[f'{mode}_loader'].get_iterator()
        losses, ys_true, ys_pred = [], [], []
        for x, y in data_iter:
            x, x_cov, x_his, y, y_cov, y_his = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask = model(x, y_cov, x_cov=x_cov, x_his=x_his, y_his=y_his)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = masked_mae_loss(y_pred, y_true)
            if args.method == "baseline":
                separate_loss = nn.TripletMarginLoss(margin=1.0)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
            elif args.method == "SCL":
                separate_loss = InfoNCELoss(temp=args.temp, contra_denominator=args.contra_denominator)
                loss2 = separate_loss(pos, neg, mask)
            else:
                pass
            compact_loss = nn.MSELoss()
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        y_size = data[f'y_{mode}'].shape[0]
        ys_true, ys_pred = torch.cat(ys_true, dim=0)[:y_size], torch.cat(ys_pred, dim=0)[:y_size]

        if mode == 'test':
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            mae = masked_mae_loss(ys_pred, ys_true).item()
            mape = masked_mape_loss(ys_pred, ys_true).item()
            rmse = masked_rmse_loss(ys_pred, ys_true).item()
            mae_3 = masked_mae_loss(ys_pred[2:3], ys_true[2:3]).item()
            mape_3 = masked_mape_loss(ys_pred[2:3], ys_true[2:3]).item()
            rmse_3 = masked_rmse_loss(ys_pred[2:3], ys_true[2:3]).item()
            mae_6 = masked_mae_loss(ys_pred[5:6], ys_true[5:6]).item()
            mape_6 = masked_mape_loss(ys_pred[5:6], ys_true[5:6]).item()
            rmse_6 = masked_rmse_loss(ys_pred[5:6], ys_true[5:6]).item()
            mae_12 = masked_mae_loss(ys_pred[11:12], ys_true[11:12]).item()
            mape_12 = masked_mape_loss(ys_pred[11:12], ys_true[11:12]).item()
            rmse_12 = masked_rmse_loss(ys_pred[11:12], ys_true[11:12]).item()
            logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae, mape, rmse))
            logger.info('Horizon 15mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_3, mape_3, rmse_3))
            logger.info('Horizon 30mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_6, mape_6, rmse_6))
            logger.info('Horizon 60mins: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mae_12, mape_12, rmse_12))
            ys_true, ys_pred = ys_true.permute(1, 0, 2, 3), ys_pred.permute(1, 0, 2, 3)
            
        return mean_loss, ys_true, ys_pred
        
def traintest_model():  
    model = get_model()
    print_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader'].get_iterator()
        losses, mae_losses, contra_losses, compact_losses = [], [], [], []
        for x, y in data_iter:
            optimizer.zero_grad()
            x, x_cov, x_his, y, y_cov, y_his = prepare_x_y(x, y)
            output, h_att, query, pos, neg, mask = model(x, y_cov, y, x_cov=x_cov, x_his=x_his, y_his=y_his, batches_seen=batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            loss1 = masked_mae_loss(y_pred, y_true)
            if args.method == "baseline":
                separate_loss = nn.TripletMarginLoss(margin=1.0)
                loss2 = separate_loss(query, pos.detach(), neg.detach())
            elif args.method == "SCL":
                separate_loss = InfoNCELoss(temp=args.temp, contra_denominator=args.contra_denominator)
                loss2 = separate_loss(pos, neg, mask)
            else:
                pass
            compact_loss = nn.MSELoss()
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            mae_losses.append(loss1.item())
            contra_losses.append(loss2.item())
            compact_losses.append(loss3.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
        train_loss = np.mean(losses)
        train_mae_loss = np.mean(mae_losses)
        train_contra_loss = np.mean(contra_losses)
        train_compact_loss = np.mean(compact_losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, train_mae_loss: {:.4f}, train_contra_loss: {:.4f}, train_compact_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, train_mae_loss, train_contra_loss, train_compact_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        test_loss, _, _ = evaluate(model, 'test')
        
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
    
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    test_loss, _, _ = evaluate(model, 'test')

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['METRLA', 'PEMSBAY'], default='METRLA', help='which dataset to run')
parser.add_argument('--trainval_ratio', type=float, default=0.8, help='the ratio of training and validation data among the total')
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--num_nodes', type=int, default=207, help='num_nodes')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
parser.add_argument('--horizon', type=int, default=12, help='output sequence length')
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--embed_dim', type=int, default=8, help='embedding dimension for adaptive graph')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K')
parser.add_argument('--rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units')
parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes')
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes')
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument('--lamb', type=float, default=0.01, help='lamb value for separate loss')
parser.add_argument('--lamb1', type=float, default=0.01, help='lamb1 value for compact loss')
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="base learning rate")
parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio")
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm")
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning")
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps")
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed')
# TODO: add more arguments for supervised contrastive learning
parser.add_argument('--delta', type=float, default=10.0, help='abnormal threshold')
parser.add_argument('--method', type=str, choices=["baseline", "SCL"], default="SCL", help='whether to use baseline or supervised contrastive learning')
parser.add_argument('--contra_denominator', action="store_false", help='abnormal threshold')
parser.add_argument('--temp', type=float, default=0.1, help='temperature')
args = parser.parse_args()
        
if args.dataset == 'METRLA':
    data_path = f'../{args.dataset}/metr-la.h5'
    args.num_nodes = 207
elif args.dataset == 'PEMSBAY':
    data_path = f'../{args.dataset}/pems-bay.h5'
    args.num_nodes = 325
else:
    pass # including more datasets in the future    

model_name = 'MemGCRN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
# score_path = f'{path}/{model_name}_{timestring}_scores.txt'
# epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
# shutil.copy2(sys.argv[0], path)
# shutil.copy2(f'{model_name}.py', path)
# shutil.copy2('utils.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('trainval_ratio', args.trainval_ratio)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('rnn_layers', args.rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('embed_dim', args.embed_dim)
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('mem_num', args.mem_num)
logger.info('mem_dim', args.mem_dim)
logger.info('loss', args.loss)
logger.info('separate loss lamb', args.lamb)
logger.info('compact loss lamb1', args.lamb1)
logger.info('batch_size', args.batch_size)
logger.info('epochs', args.epochs)
logger.info('patience', args.patience)
logger.info('lr', args.lr)
logger.info('epsilon', args.epsilon)
logger.info('steps', args.steps)
logger.info('lr_decay_ratio', args.lr_decay_ratio)
logger.info('use_curriculum_learning', args.use_curriculum_learning)
# TODO: add more arguments for supervised contrastive learning
logger.info('delta', args.delta)
logger.info('method', args.method)
logger.info('contra_denominator', args.contra_denominator)
logger.info('temp', args.temp)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
# Please comment the following three lines for running experiments multiple times.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
#####################################################################################################

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../{args.dataset}', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
data['train_loader'] = DataLoader(data['x_train'], data['y_train'], args.batch_size, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], args.batch_size, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], args.batch_size, shuffle=False)

def main():
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()