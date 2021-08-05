import pandas as pd
import numpy as np
import random
from rstl import STL
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import argparse
import os

import time

# hyper-parameters
parser = argparse.ArgumentParser(description='MuSDRI algorithm on GAS dataset')
parser.add_argument(
    '--gpu', default=0, type=int,
    help='number of gpu')  # -1 = all GPUs, 0 = 1080Ti 1, 1 = 1080
parser.add_argument('--seq_len', default=72, type=int, help='sequence length')
parser.add_argument('--num_features',
                    default=5,
                    type=int,
                    help='number of features for input dataset ')
parser.add_argument('--rnn_hid_size',
                    default=128,
                    type=int,
                    help='number of hidden size for RNN ')
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')
parser.add_argument('--frequence',
                    default=[4, 12, 84],
                    type=list,
                    help='weight for regularization term (lambda)')
parser.add_argument('--WRT',
                    default=1.0,
                    type=float,
                    help='weight for regularization term (lambda)')

args = parser.parse_args()

# gpu settings
if args.gpu != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = SummaryWriter()

seq_len = args.seq_len
frequence1, frequence2, frequence3 = args.frequence


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TemporalDecay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert (input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * self.m, self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma


class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))

        m = torch.ones(input_size, input_size) - torch.eye(
            input_size, input_size)
        self.register_buffer('m', m)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * self.m, self.b)
        return z_h


class weight_sumone_linear(nn.Module):
    def __init__(self, input_size, output_size):
        super(weight_sumone_linear, self).__init__()
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))

    def get_weights(self):
        return F.softmax(self.W)

    def forward(self, x):
        x = torch.swapdims(x, 1, 2)
        w_normalized = F.softmax(self.W)
        x = F.linear(x, w_normalized)
        output = torch.swapdims(x, 1, 2)
        return output


class rits(nn.Module):
    def __init__(self, rnn_hid_size):
        super(rits, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(args.num_features * 2, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size=args.num_features,
                                          output_size=self.rnn_hid_size,
                                          diag=False)
        self.temp_decay_x = TemporalDecay(input_size=args.num_features,
                                          output_size=args.num_features,
                                          diag=True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, args.num_features)
        self.feat_reg = FeatureRegression(args.num_features)

        self.weight_combine = nn.Linear(args.num_features * 2,
                                        args.num_features)

        self.dropout = nn.Dropout(p=0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)

        self.combine = weight_sumone_linear(3, 1)

    def forward(self, data, direct):

        if direct == 'forward':
            originals = data[:, 0, :, :]
            masks = data[:, 1, :, :]
            deltas = data[:, 2, :, :]

            remainder1 = data[:, 3, :, :]
            trend1 = data[:, 4, :, :]
            seasonal1 = data[:, 5, :, :]

            remainder2 = data[:, 6, :, :]
            trend2 = data[:, 7, :, :]
            seasonal2 = data[:, 8, :, :]

            remainder3 = data[:, 9, :, :]
            trend3 = data[:, 10, :, :]
            seasonal3 = data[:, 11, :, :]

        if direct == 'backward':
            originals = data[:, 12, :, :]
            masks = data[:, 13, :, :]
            deltas = data[:, 14, :, :]

            remainder1 = data[:, 15, :, :]
            trend1 = data[:, 16, :, :]
            seasonal1 = data[:, 17, :, :]

            remainder2 = data[:, 18, :, :]
            trend2 = data[:, 19, :, :]
            seasonal2 = data[:, 20, :, :]

            remainder3 = data[:, 21, :, :]
            trend3 = data[:, 22, :, :]
            seasonal3 = data[:, 23, :, :]

        h = torch.zeros((originals.size()[0], 3, self.rnn_hid_size))
        c = torch.zeros((originals.size()[0], 3, self.rnn_hid_size))

        h, c = h.to(device), c.to(device)

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(data.shape[2]):
            m = masks[:, t, :]
            d = deltas[:, t, :]
            m = torch.cat([m.unsqueeze(1),
                           m.unsqueeze(1),
                           m.unsqueeze(1)],
                          dim=1)
            d = torch.cat([d.unsqueeze(1),
                           d.unsqueeze(1),
                           d.unsqueeze(1)],
                          dim=1)

            r1 = remainder1[:, t, :]
            r2 = remainder2[:, t, :]
            r3 = remainder3[:, t, :]
            r = torch.cat([r1.unsqueeze(1),
                           r2.unsqueeze(1),
                           r3.unsqueeze(1)],
                          dim=1)

            t1 = trend1[:, t, :]
            t2 = trend2[:, t, :]
            t3 = trend3[:, t, :]
            tr = torch.cat([t1.unsqueeze(1),
                            t2.unsqueeze(1),
                            t3.unsqueeze(1)],
                           dim=1)

            s1 = seasonal1[:, t, :]
            s2 = seasonal2[:, t, :]
            s3 = seasonal3[:, t, :]
            s = torch.cat([s1.unsqueeze(1),
                           s2.unsqueeze(1),
                           s3.unsqueeze(1)],
                          dim=1)

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)

            x_loss += torch.sum(self.combine(
                torch.abs(r - x_h) * m)) / (torch.sum(self.combine(m)) + 1e-5)

            x_c = m * r + (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(self.combine(
                torch.abs(r - z_h) * m)) / (torch.sum(self.combine(m)) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim=2))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(self.combine(
                torch.abs(r - c_h) * m)) / (torch.sum(self.combine(m)) + 1e-5)

            w_normalized = self.combine.get_weights()
            loss_weight = w_normalized * torch.tensor(
                [1 / frequence1, 1 / frequence2, 1 / frequence3]).cuda()
            x_loss += args.WRT * loss_weight.sum()

            c_c = m * r + (1 - m) * c_h

            inputs = torch.cat([c_c, m], dim=2)

            inputs = inputs.reshape(-1, inputs.shape[-1])
            h = h.reshape(-1, h.shape[-1])
            c = c.reshape(-1, c.shape[-1])

            h = self.dropout(h)

            h, c = self.rnn_cell(inputs, (h, c))

            h = h.reshape(-1, 3, h.shape[-1])
            c = c.reshape(-1, 3, c.shape[-1])

            impute = self.combine(c_c + s + tr).reshape(data.shape[0], -1)

            imputations.append(impute.unsqueeze(dim=1))

        imputations = torch.cat(imputations, dim=1)

        return {'loss': x_loss, 'imputations': imputations}

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data, direct='forward')

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()
        return ret


class Brits(nn.Module):
    def __init__(self, rnn_hid_size):
        super(Brits, self).__init__()

        self.rnn_hid_size = rnn_hid_size

        self.build()

    def build(self):
        self.rits_f = rits(self.rnn_hid_size)
        self.rits_b = rits(self.rnn_hid_size)

    def forward(self, data):
        ret_f = self.rits_f(data, 'forward')
        ret_b = self.reverse(self.rits_b(data, 'backward'))

        ret = self.merge_ret(ret_f, ret_b)

        return ret

    def merge_ret(self, ret_f, ret_b):
        loss_f = ret_f['loss']
        loss_b = ret_b['loss']
        loss_c = self.get_consistency_loss(ret_f['imputations'],
                                           ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

        log.add_scalars(
            "weights/season_f",
            {str(k): v
             for k, v in enumerate(self.rits_f.combine.W[0])})
        log.add_scalars(
            "weights/season_b",
            {str(k): v
             for k, v in enumerate(self.rits_b.combine.W[0])})
        log.add_scalars("weights/season_softmax_f", {
            str(k): v
            for k, v in enumerate(self.rits_f.combine.get_weights()[0])
        })
        log.add_scalars("weights/season_softmax_b", {
            str(k): v
            for k, v in enumerate(self.rits_b.combine.get_weights()[0])
        })
        mode = 'train' if self.training else 'test'
        log.add_scalar(f"{mode}_loss/loss_f", loss_f)
        log.add_scalar(f"{mode}_loss/loss_b", loss_b)
        log.add_scalar(f"{mode}_loss/loss_c", loss_c)
        log.add_scalar(f"{mode}_loss/loss", loss)

        imputations = (ret_f['imputations'] + ret_b['imputations']) / 2

        ret_f['loss'] = loss
        ret_f['imputations'] = imputations

        return ret_f

    def get_consistency_loss(self, pred_f, pred_b):
        loss = torch.abs(pred_f - pred_b).mean() * 1e-1
        return loss

    def reverse(self, ret):
        def reverse_tensor(tensor_):
            if tensor_.dim() <= 1:
                return tensor_
            indices = range(tensor_.size()[1])[::-1]
            indices = Variable(torch.LongTensor(indices), requires_grad=False)

            if torch.cuda.is_available():
                indices = indices.cuda()

            return tensor_.index_select(1, indices)

        for key in ret:
            ret[key] = reverse_tensor(ret[key])

        return ret

    def run_on_batch(self, data, optimizer, epoch=None):
        ret = self(data)

        if optimizer is not None:
            optimizer.zero_grad()
            ret['loss'].backward()
            optimizer.step()

        return ret


def incision(data, length):
    new_dataset = []
    for i in range(len(data) - length + 1):
        new_dataset.append(data[i:i + length])
    new_dataset = np.array(new_dataset)
    new_dataset = new_dataset.reshape(new_dataset.shape[0], 1,
                                      new_dataset.shape[1],
                                      new_dataset.shape[2])
    return new_dataset


def delta_func(mask_dataframe):
    deltas_matrix = pd.DataFrame(
        columns=mask_dataframe.columns.values.tolist())
    for col in mask_dataframe.columns.values.tolist():
        mask = mask_dataframe[col]
        deltas = [0]
        for i in range(1, len(mask)):
            if mask[i - 1] == 1:
                deltas.append(1)
            elif mask[i - 1] == 0:
                deltas.append(1 + deltas[i - 1])
        deltas_matrix[col] = deltas

    return np.array(deltas_matrix)


if not os.path.exists('./losses/'):
    os.makedirs('./losses/')
if not os.path.exists('./checkpoint/'):
    os.makedirs('./checkpoint/')

# Read training set
train_data_statiton = pd.read_csv('./data/train/train_normal.csv')
train_data_null = pd.read_csv('./data/train/train_null.csv')
train_mask = pd.read_csv('./data/train/train_mask.csv')
train_org_mask = pd.read_csv('data/train/train_org_mask.csv')

train_remainder1 = pd.read_csv('./data/train/stlplus_train_remainder' +
                               str(frequence1) + '.csv',
                               index_col=0)
train_remainder2 = pd.read_csv('./data/train/stlplus_train_remainder' +
                               str(frequence2) + '.csv',
                               index_col=0)
train_remainder3 = pd.read_csv('./data/train/stlplus_train_remainder' +
                               str(frequence3) + '.csv',
                               index_col=0)

train_seasonal1 = pd.read_csv('./data/train/stlplus_train_seasonal' +
                              str(frequence1) + '.csv',
                              index_col=0)
train_seasonal2 = pd.read_csv('./data/train/stlplus_train_seasonal' +
                              str(frequence2) + '.csv',
                              index_col=0)
train_seasonal3 = pd.read_csv('./data/train/stlplus_train_seasonal' +
                              str(frequence3) + '.csv',
                              index_col=0)

train_trend1 = pd.read_csv('./data/train/stlplus_train_trend' +
                           str(frequence1) + '.csv',
                           index_col=0)
train_trend2 = pd.read_csv('./data/train/stlplus_train_trend' +
                           str(frequence2) + '.csv',
                           index_col=0)
train_trend3 = pd.read_csv('./data/train/stlplus_train_trend' +
                           str(frequence3) + '.csv',
                           index_col=0)

train_dataset_value = train_data_null.to_numpy()
train_dataset_value[np.isnan(train_dataset_value)] = 0
train_dataset_value = incision(train_dataset_value, seq_len)

train_allnull_mask = train_mask.to_numpy() * train_org_mask.to_numpy()
train_dataset_mask = train_allnull_mask
train_dataset_mask = incision(train_dataset_mask, seq_len)

train_allnull_mask = pd.DataFrame(train_allnull_mask)
train_dataset_deltas = delta_func(train_allnull_mask)
train_dataset_deltas = incision(train_dataset_deltas, seq_len)

train_remainder1 = train_remainder1.to_numpy() * train_org_mask.to_numpy(
) * train_mask.to_numpy()
train_remainder1[np.isnan(train_remainder1)] = 0
train_remainder1 = incision(train_remainder1, seq_len)

train_trend1 = incision(train_trend1.to_numpy(), seq_len)

train_seasonal1 = incision(train_seasonal1.to_numpy(), seq_len)

train_remainder2 = train_remainder2.to_numpy() * train_org_mask.to_numpy(
) * train_mask.to_numpy()
train_remainder2[np.isnan(train_remainder2)] = 0
train_remainder2 = incision(train_remainder2, seq_len)

train_trend2 = incision(train_trend2.to_numpy(), seq_len)

train_seasonal2 = incision(train_seasonal2.to_numpy(), seq_len)

train_remainder3 = train_remainder3.to_numpy() * train_org_mask.to_numpy(
) * train_mask.to_numpy()
train_remainder3[np.isnan(train_remainder3)] = 0
train_remainder3 = incision(train_remainder3, seq_len)

train_trend3 = incision(train_trend3.to_numpy(), seq_len)

train_seasonal3 = incision(train_seasonal3.to_numpy(), seq_len)

train_dataset = np.concatenate(
    (train_dataset_value, train_dataset_mask, train_dataset_deltas,
     train_remainder1, train_trend1, train_seasonal1, train_remainder2,
     train_trend2, train_seasonal2, train_remainder3, train_trend3,
     train_seasonal3, train_dataset_value[:, :, ::-1, :],
     train_dataset_mask[:, :, ::-1, :], train_dataset_deltas[:, :, ::-1, :],
     train_remainder1[:, :, ::-1, :], train_trend1[:, :, ::-1, :],
     train_seasonal1[:, :, ::-1, :], train_remainder2[:, :, ::-1, :],
     train_trend2[:, :, ::-1, :], train_seasonal2[:, :, ::-1, :],
     train_remainder3[:, :, ::-1, :], train_trend3[:, :, ::-1, :],
     train_seasonal3[:, :, ::-1, :]), 1)

# read test set
test_data_statiton = pd.read_csv('data/test/test_normal.csv')
test_data_null = pd.read_csv('data/test/test_null.csv')
test_mask = pd.read_csv('data/test/test_mask.csv')
test_org_mask = pd.read_csv('data/test/test_org_mask.csv')

test_remainder1 = pd.read_csv('./data/test/stlplus_test_remainder' +
                              str(frequence1) + '.csv',
                              index_col=0)
test_remainder2 = pd.read_csv('./data/test/stlplus_test_remainder' +
                              str(frequence2) + '.csv',
                              index_col=0)
test_remainder3 = pd.read_csv('./data/test/stlplus_test_remainder' +
                              str(frequence3) + '.csv',
                              index_col=0)

test_seasonal1 = pd.read_csv('./data/test/stlplus_test_seasonal' +
                             str(frequence1) + '.csv',
                             index_col=0)
test_seasonal2 = pd.read_csv('./data/test/stlplus_test_seasonal' +
                             str(frequence2) + '.csv',
                             index_col=0)
test_seasonal3 = pd.read_csv('./data/test/stlplus_test_seasonal' +
                             str(frequence3) + '.csv',
                             index_col=0)

test_trend1 = pd.read_csv('./data/test/stlplus_test_trend' + str(frequence1) +
                          '.csv',
                          index_col=0)
test_trend2 = pd.read_csv('./data/test/stlplus_test_trend' + str(frequence2) +
                          '.csv',
                          index_col=0)
test_trend3 = pd.read_csv('./data/test/stlplus_test_trend' + str(frequence3) +
                          '.csv',
                          index_col=0)

test_null_number = test_data_null.isna().sum().sum() - test_data_statiton.isna(
).sum().sum()

test_dataset_value = test_data_null.to_numpy()
test_dataset_value[np.isnan(test_dataset_value)] = 0
test_dataset_value = incision(test_dataset_value, len(test_data_statiton))

test_allnull_mask = test_mask.to_numpy() * test_org_mask.to_numpy()
test_dataset_mask = test_allnull_mask
test_dataset_mask = incision(test_dataset_mask, len(test_data_statiton))

test_allnull_mask = pd.DataFrame(test_allnull_mask)
test_dataset_deltas = delta_func(test_allnull_mask)
test_dataset_deltas = incision(test_dataset_deltas, len(test_data_statiton))

test_remainder1 = test_remainder1.to_numpy() * test_org_mask.to_numpy(
) * test_mask.to_numpy()
test_remainder1[np.isnan(test_remainder1)] = 0
test_remainder1 = incision(test_remainder1, len(test_data_statiton))

test_trend1 = incision(test_trend1.to_numpy(), len(test_data_statiton))

test_seasonal1 = incision(test_seasonal1.to_numpy(), len(test_data_statiton))

test_remainder2 = test_remainder2.to_numpy() * test_org_mask.to_numpy(
) * test_mask.to_numpy()
test_remainder2[np.isnan(test_remainder2)] = 0
test_remainder2 = incision(test_remainder2, len(test_data_statiton))

test_trend2 = incision(test_trend2.to_numpy(), len(test_data_statiton))

test_seasonal2 = incision(test_seasonal2.to_numpy(), len(test_data_statiton))

test_remainder3 = test_remainder3.to_numpy() * test_org_mask.to_numpy(
) * test_mask.to_numpy()
test_remainder3[np.isnan(test_remainder3)] = 0
test_remainder3 = incision(test_remainder3, len(test_data_statiton))

test_trend3 = incision(test_trend3.to_numpy(), len(test_data_statiton))

test_seasonal3 = incision(test_seasonal3.to_numpy(), len(test_data_statiton))

test_dataset = np.concatenate(
    (test_dataset_value, test_dataset_mask, test_dataset_deltas,
     test_remainder1, test_trend1, test_seasonal1, test_remainder2,
     test_trend2, test_seasonal2, test_remainder3, test_trend3, test_seasonal3,
     test_dataset_value[:, :, ::-1, :], test_dataset_mask[:, :, ::-1, :],
     test_dataset_deltas[:, :, ::-1, :], test_remainder1[:, :, ::-1, :],
     test_trend1[:, :, ::-1, :], test_seasonal1[:, :, ::-1, :],
     test_remainder2[:, :, ::-1, :], test_trend2[:, :, ::-1, :],
     test_seasonal2[:, :, ::-1, :], test_remainder3[:, :, ::-1, :],
     test_trend3[:, :, ::-1, :], test_seasonal3[:, :, ::-1, :]), 1)

trainloader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=64,
                                          shuffle=True,
                                          num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=1,
                                         shuffle=False,
                                         num_workers=4)

model = Brits(rnn_hid_size=args.rnn_hid_size)
for p in model.parameters():
    torch.nn.init.normal_(p.data, std=0.01)
model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

lr_milestones = [20, 40]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                           milestones=lr_milestones,
                                           gamma=0.1)
best_loss = float('inf')

for epoch in range(10):
    scheduler.step()

    for index, data in enumerate(trainloader):
        model.train()
        total_loss_train = AverageMeter()
        data = data.to(device).float()
        ret = model.run_on_batch(data, optimizer, epoch)
        total_loss_train.update(ret['loss'].item(), data.size(0))
        log.add_scalar('loss/train_loss', total_loss_train.avg)

        # evaluate
        if index % 10 == 0:
            print('Epoch: {}, index: {}'.format(epoch, int(index)))
            print('train loss: {:.4f}'.format(total_loss_train.avg))

            # evaluate
            model.eval()
            test_len = len(test_data_statiton)
            for i, data in enumerate(testloader):
                data = data.to(device).float()
                ret = model.run_on_batch(data, None)
                test_loss = ret['loss'].item()
                Y_predicted = ret['imputations'].detach().cpu().numpy()
                # print(ret['imputations'].shape, Y_predicted.shape, test_mask.shape)
                Y_predicted = np.squeeze(Y_predicted)

            print('test loss: {:.4f}'.format(test_loss))
            log.add_scalar('loss/test_loss', test_loss)

            error_mask = (test_data_statiton.fillna(0).to_numpy() -
                          Y_predicted) * (1 - test_mask).to_numpy()

            mse_error = error_mask**2
            mae_error = mre_error = abs(error_mask)

            total_error_MSE = mse_error.sum().sum()
            total_error_MAE = mae_error.sum().sum()
            total_error_MRE = mre_error.sum().sum()

            total_label_MRE = abs(
                test_data_statiton.fillna(0).to_numpy() *
                (1 - test_mask).to_numpy()).sum().sum()

            mse = total_error_MSE / test_null_number
            mae = total_error_MAE / test_null_number
            mre = total_error_MRE / total_label_MRE
            print('mse: {:.3f}, mae: {:.3f}, mre: {:.3f}'.format(
                mse, mae, mre))
            log.add_scalars('eval', {'mse': mse, 'mae': mae, 'mre': mre})

            file = open("./losses/MuSDRI.txt", "a")
            file.write("Epoch = {}, index = {}  ".format(epoch, int(index)))
            file.write("\n")

            file.write("trainloss = {:.3f}  ".format(total_loss_train.avg))
            file.write("\n")

            file.write("testloss = {:.3f}   ".format(test_loss))
            file.write("\n")

            file.write("testmse = {:.4f}   ".format(mse))
            file.write("\n")

            file.write("testmae = {:.4f}   ".format(mae))
            file.write("\n")

            file.write("testmre = {:.4f}   ".format(mre))
            file.write("\n")

            file.close

            if test_loss < best_loss:
                best_loss = test_loss
                model_state = {
                    'net_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'index': index,
                    'best_loss': best_loss
                }

                save_point = './checkpoint/MuSDRI'
                torch.save(model_state, save_point)

print('finished training')
checkpoint = torch.load('./checkpoint/MuSDRI')
model.load_state_dict(checkpoint['net_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
index = checkpoint['index']
test_loss = checkpoint['best_loss']

print('Best model: Epoch: {}, index: {}, best_loss = {:.4f}'.format(
    epoch, index, test_loss))

model.eval()
for index, data in enumerate(testloader):
    data = data.to(device).float()
    ret = model.run_on_batch(data, None)
    Y_predicted = ret['imputations'].data.cpu().numpy()
    Y_predicted = np.squeeze(Y_predicted)

error_mask = (test_data_statiton.fillna(0).to_numpy() -
              Y_predicted) * (1 - test_mask).to_numpy()

mse_error = error_mask**2
mae_error = mre_error = abs(error_mask)

total_error_MSE = mse_error.sum().sum()
total_error_MAE = mae_error.sum().sum()
total_error_MRE = mre_error.sum().sum()

total_label_MRE = abs(
    test_data_statiton.fillna(0).to_numpy() *
    (1 - test_mask).to_numpy()).sum().sum()

mse = total_error_MSE / test_null_number
mae = total_error_MAE / test_null_number
mre = total_error_MRE / total_label_MRE

file = open('results_MAE_MRE_MSE.txt', 'a')
file.write('MuSDRI\n')
file.write('seq_len = {}, frequence = {},  WRT = {} '.format(
    seq_len, args.frequence, args.WRT))
file.write('\n')
file.write('MSE = {:.5f}   '.format(mse))
file.write('MAE = {:.5f}   '.format(mae))
file.write('MRE = {:.5f}   '.format(mre))
file.write('\n')
file.close

print(mse)
print(mae)
print(mre)

#save imputation data
imputed_data = pd.DataFrame(Y_predicted)
imputed_data.columns = test_data_statiton.columns.tolist()
imputed_data.to_csv('./data/test/impute/MuSDRI.csv', index=0)
print('Finish saving imputation data')
