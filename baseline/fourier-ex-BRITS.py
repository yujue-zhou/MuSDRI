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

import argparse
import os

import time

# hyper-parameters
parser = argparse.ArgumentParser(description='fourier-exogenous-BRITS algorithm on GAS dataset')
parser.add_argument('--gpu', default=0, type=int, help='number of gpu') # -1 = all GPUs, 0 = 1080Ti 1, 1 = 1080
parser.add_argument('--seq_len', default=36, type=int, help='sequence length') 
parser.add_argument('--num_features', default=5, type=int, help='number of features for input dataset ') 
parser.add_argument('--rnn_hid_size', default=128, type=int, help='number of hidden size for RNN ') 
parser.add_argument('--lr', default=0.001, type=float, help='learning_rate')

args = parser.parse_args()

# gpu settings
if args.gpu != -1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_len = args.seq_len

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
    def __init__(self, input_size, output_size, diag = False):
        super(TemporalDecay, self).__init__()
        self.diag = diag

        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
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

        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
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


class rits(nn.Module):
    def __init__(self, rnn_hid_size):
        super(rits, self).__init__()

        self.rnn_hid_size = rnn_hid_size
        self.build()

    def build(self):
        self.rnn_cell = nn.LSTMCell(args.num_features*4, self.rnn_hid_size)

        self.temp_decay_h = TemporalDecay(input_size = args.num_features, output_size = self.rnn_hid_size, diag = False)
        self.temp_decay_x = TemporalDecay(input_size = args.num_features, output_size = args.num_features, diag = True)

        self.hist_reg = nn.Linear(self.rnn_hid_size, args.num_features)
        self.feat_reg = FeatureRegression(args.num_features)

        self.weight_combine = nn.Linear(args.num_features*2, args.num_features)

        self.dropout = nn.Dropout(p = 0.25)
        self.out = nn.Linear(self.rnn_hid_size, 1)


    def forward(self, data, direct):

        if direct=='forward':
          values = data[:,0,:,:]
          masks = data[:,1,:,:]
          deltas = data[:,2,:,:]

          fourier_s1 = data[:,3,:,:]
          fourier_s2 = data[:,4,:,:]

        if direct=='backward':
          values = data[:,5,:,:]
          masks = data[:,6,:,:]
          deltas = data[:,7,:,:]

          fourier_s1 = data[:,8,:,:]
          fourier_s2 = data[:,9,:,:]

        h = torch.zeros((values.size()[0], self.rnn_hid_size))
        c = torch.zeros((values.size()[0], self.rnn_hid_size))

        h, c = h.to(device), c.to(device)

        x_loss = 0.0
        y_loss = 0.0

        imputations = []

        for t in range(data.shape[2]):
            x = values[:, t, :]
            m = masks[:, t, :]
            d = deltas[:, t, :]

            s1 = fourier_s1[:, t, :]
            s2 = fourier_s2[:, t, :]

            gamma_h = self.temp_decay_h(d)
            gamma_x = self.temp_decay_x(d)

            h = h * gamma_h

            x_h = self.hist_reg(h)
            x_loss += torch.sum(torch.abs(x - x_h) * m) / (torch.sum(m) + 1e-5)

            x_c =  m * x +  (1 - m) * x_h

            z_h = self.feat_reg(x_c)
            x_loss += torch.sum(torch.abs(x - z_h) * m) / (torch.sum(m) + 1e-5)

            alpha = self.weight_combine(torch.cat([gamma_x, m], dim = 1))

            c_h = alpha * z_h + (1 - alpha) * x_h
            x_loss += torch.sum(torch.abs(x - c_h) * m) / (torch.sum(m) + 1e-5)

            c_c = m * x + (1 - m) * c_h

            inputs = torch.cat([c_c, m, s1, s2], dim = 1)

            h, c = self.rnn_cell(inputs, (h, c))

            imputations.append(c_c.unsqueeze(dim = 1))


        imputations = torch.cat(imputations, dim = 1)

        return {'loss': x_loss ,'imputations': imputations}

    def run_on_batch(self, data, optimizer, epoch = None):
        ret = self(data, direct = 'forward')

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
        loss_c = self.get_consistency_loss(ret_f['imputations'], ret_b['imputations'])

        loss = loss_f + loss_b + loss_c

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
            indices = Variable(torch.LongTensor(indices), requires_grad = False)

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
  new_dataset=[]
  for i in range(len(data)-length+1):
    new_dataset.append(data[i:i+length])
  new_dataset = np.array(new_dataset)
  new_dataset = new_dataset.reshape(new_dataset.shape[0],1,new_dataset.shape[1],new_dataset.shape[2])
  return new_dataset


def delta_func(mask_dataframe):
  deltas_matrix = pd.DataFrame(columns=mask_dataframe.columns.values.tolist())
  for col in mask_dataframe.columns.values.tolist():
    mask = mask_dataframe[col]
    deltas = [0]
    for i in range(1,len(mask)):
      if mask[i-1]==1:
        deltas.append(1)
      elif mask[i-1]==0:
        deltas.append(1+deltas[i-1])
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

train_seasonality1 = pd.read_csv('./data/train/fourier_seasonality1.csv', index_col=0)
train_seasonality2 = pd.read_csv('./data/train/fourier_seasonality2.csv', index_col=0)


train_dataset_value= train_data_null.to_numpy()
train_dataset_value[np.isnan(train_dataset_value)]=0
train_dataset_value = incision(train_dataset_value, seq_len)

train_allnull_mask = train_mask.to_numpy()*train_org_mask.to_numpy()
train_dataset_mask= train_allnull_mask
train_dataset_mask = incision(train_dataset_mask, seq_len)

train_allnull_mask = pd.DataFrame(train_allnull_mask)
train_dataset_deltas= delta_func(train_allnull_mask)
train_dataset_deltas = incision(train_dataset_deltas, seq_len)


train_seasonality1 = train_seasonality1.to_numpy()
train_seasonality1 = incision(train_seasonality1, seq_len)

train_seasonality2 = train_seasonality2.to_numpy()
train_seasonality2 = incision(train_seasonality2, seq_len)

train_dataset = np.concatenate((train_dataset_value,train_dataset_mask,train_dataset_deltas, 
  train_seasonality1, train_seasonality2, 
  train_dataset_value[:,:,::-1,:], train_dataset_mask[:,:,::-1,:], train_dataset_deltas[:,:,::-1,:],
  train_seasonality1[:,:,::-1,:], train_seasonality2[:,:,::-1,:]
  ),1)

# read test set
test_data_statiton = pd.read_csv('data/test/test_normal.csv')
test_data_null = pd.read_csv('data/test/test_null.csv')
test_mask = pd.read_csv('data/test/test_mask.csv')
test_org_mask = pd.read_csv('data/test/test_org_mask.csv')

test_trend = pd.read_csv('./data/test/mstlplus_trend.csv', index_col=0)
test_seasonality1 = pd.read_csv('./data/test/fourier_seasonality1.csv', index_col=0)
test_seasonality2 = pd.read_csv('./data/test/fourier_seasonality2.csv', index_col=0)


test_null_number =test_data_null.isna().sum().sum()-test_data_statiton.isna().sum().sum()


test_dataset_value= test_data_null.to_numpy()
test_dataset_value[np.isnan(test_dataset_value)]=0
test_dataset_value = incision(test_dataset_value, len(test_data_statiton))

test_allnull_mask = test_mask.to_numpy()*test_org_mask.to_numpy()
test_dataset_mask= test_allnull_mask
test_dataset_mask = incision(test_dataset_mask, len(test_data_statiton))

test_allnull_mask = pd.DataFrame(test_allnull_mask)
test_dataset_deltas= delta_func(test_allnull_mask)
test_dataset_deltas = incision(test_dataset_deltas, len(test_data_statiton))

test_seasonality1 = test_seasonality1.to_numpy()
test_seasonality1 = incision(test_seasonality1, len(test_data_statiton))

test_seasonality2 = test_seasonality2.to_numpy()
test_seasonality2 = incision(test_seasonality2, len(test_data_statiton))


test_dataset = np.concatenate((test_dataset_value,test_dataset_mask,test_dataset_deltas, 
  test_seasonality1, test_seasonality2,
  test_dataset_value[:,:,::-1,:], test_dataset_mask[:,:,::-1,:], test_dataset_deltas[:,:,::-1,:],
  test_seasonality1[:,:,::-1,:], test_seasonality2[:,:,::-1,:]
  ),1)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)


model = Brits(rnn_hid_size=args.rnn_hid_size)
model.to(device)

optimizer = optim.Adam(params=model.parameters(), lr=args.lr)

lr_milestones = [20, 40]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)
best_loss = float('inf')

for epoch in range(20):
  scheduler.step()
  model.train()
  total_loss_train = AverageMeter()

  for index, data in enumerate(trainloader):
    data = data.to(device).float()
    ret = model.run_on_batch(data, optimizer, epoch)
    total_loss_train.update(ret['loss'].item(), data.size(0))

    # evaluate
    if index%10==0:
      print('Epoch: {}, index: {}'.format(epoch,int(index)))
      print('train loss: {:.4f}'.format(total_loss_train.avg))
      
      # evaluate
      model.eval()
      test_len = len(test_data_statiton)
      for i, data in enumerate(testloader):
        data = data.to(device).float()
        ret = model.run_on_batch(data, None)
        test_loss = ret['loss'].item()
        Y_predicted = ret['imputations'].data.cpu().numpy()
        Y_predicted = np.squeeze(Y_predicted)

      print('test loss: {:.4f}'.format(test_loss))

      error_mask = (test_data_statiton.fillna(0).to_numpy()-Y_predicted)*(1-test_mask).to_numpy()

      mse_error = error_mask**2
      mae_error = mre_error = abs(error_mask)

      total_error_MSE = mse_error.sum().sum()
      total_error_MAE = mae_error.sum().sum()
      total_error_MRE = mre_error.sum().sum()

      total_label_MRE = abs(test_data_statiton.fillna(0).to_numpy()*(1-test_mask).to_numpy()).sum().sum()

      mse = total_error_MSE/test_null_number
      mae = total_error_MAE/test_null_number
      mre = total_error_MRE/total_label_MRE
      print('mse: {:.3f}, mae: {:.3f}, mre: {:.3f}'.format(mse, mae, mre))

      file = open("./losses/fourier-exogenous-BRITS.txt","a")
      file.write("Epoch = {}, index = {}  ".format(epoch,int(index)))
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

      if test_loss<best_loss:
        best_loss = test_loss
        model_state = {
            'net_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch':epoch,
            'index':index,
            'best_loss':best_loss
        }
       
        save_point = './checkpoint/fourier-exogenous-BRITS'
        torch.save(model_state, save_point)
        

print('finished training')
checkpoint = torch.load('./checkpoint/fourier-exogenous-BRITS')
model.load_state_dict(checkpoint['net_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
index = checkpoint['index']
test_loss = checkpoint['best_loss']

print('Best model: Epoch: {}, index: {}, best_loss = {:.4f}'.format(epoch, index, test_loss))



model.eval()
for index, data in enumerate(testloader):
  data = data.to(device).float()
  ret = model.run_on_batch(data, None)
  Y_predicted = ret['imputations'].data.cpu().numpy()
  Y_predicted = np.squeeze(Y_predicted)

error_mask = (test_data_statiton.fillna(0).to_numpy()-Y_predicted)*(1-test_mask).to_numpy()

mse_error = error_mask**2
mae_error = mre_error = abs(error_mask)

total_error_MSE = mse_error.sum().sum()
total_error_MAE = mae_error.sum().sum()
total_error_MRE = mre_error.sum().sum()

total_label_MRE = abs(test_data_statiton.fillna(0).to_numpy()*(1-test_mask).to_numpy()).sum().sum()

mse = total_error_MSE/test_null_number
mae = total_error_MAE/test_null_number
mre = total_error_MRE/total_label_MRE


file = open('results_MAE_MRE_MSE.txt','a')
file.write('fourier-exogenous-BRITS\n')
file.write('seq_len = {},   '.format(seq_len))
file.write('frequence = {12, 84}')
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
imputed_data.to_csv('./data/test/impute/fourier-exogenous-BRITS.csv',index=0)
print('Finish saving imputation data')