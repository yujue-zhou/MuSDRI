import pandas as pd
import numpy as np
import random
import time

import os



#归一化
def df_norm(df):
    columns = df.columns.tolist()
    df_n = df.copy()
    for col in columns:
        mean = df_n[col].mean()
        std = df_n[col].std()
        df_n[col] = (df_n[col] - mean) / std
    return (df_n)


file_path = './original_data/'
file_name = 'Gas_Consumption_data.csv'

all_users_data = pd.read_csv(file_path+file_name)
all_users_data.set_index(['utc_time'], inplace=True) 

if not os.path.isdir('data'):
  os.mkdir('data')

#归一化
normal_data = df_norm(all_users_data)
normal_data.to_csv('data/norm_data.csv', index = 1)


#生成随机缺失
null_rate = 0.1

orign_null_mask = all_users_data.notna()

orign_null_1d = orign_null_mask.to_numpy().flatten()
notna_index = np.where(orign_null_1d==True)[0]

random_null_mask = np.ones(len(orign_null_1d))

random_number = int(notna_index.shape[0]*null_rate)
np.random.seed(0)
random_index = np.random.choice(notna_index, size=random_number, replace=False)

for i in random_index:
  random_null_mask[i]=np.nan

random_null_mask = random_null_mask.reshape((normal_data.shape[0], normal_data.shape[1]))

data_null = normal_data*random_null_mask
data_null.to_csv('data/data_null.csv',index=1)

data_null = data_null.reset_index()
random_null_mask = pd.DataFrame(random_null_mask)
random_null_mask = random_null_mask.fillna(0)
random_null_mask['utc_time']=data_null['utc_time']
normal_data = normal_data.reset_index()
orign_null_mask = orign_null_mask.reset_index()


#save test data
test_data_normal = normal_data[(normal_data['utc_time']>='2020-01-01 00:00:00')]
test_data_null = data_null[(data_null['utc_time']>='2020-01-01 00:00:00')]
test_mask = random_null_mask[(random_null_mask['utc_time']>='2020-01-01 00:00:00')]
test_org_nullmask = orign_null_mask[(orign_null_mask['utc_time']>='2020-01-01 00:00:00')]


test_data_normal.set_index(['utc_time'], inplace=True)
test_data_null.set_index(['utc_time'], inplace=True)
test_mask.set_index(['utc_time'], inplace=True)
test_org_nullmask.set_index(['utc_time'], inplace=True)

if not os.path.isdir('data/test/'):
  os.mkdir('data/test/')
test_data_normal.to_csv('data/test/test_normal.csv',index=0)
test_data_null.to_csv('data/test/test_null.csv',index=0)
test_mask.to_csv('data/test/test_mask.csv',index=0)
test_org_nullmask.to_csv('data/test/test_org_mask.csv',index=0)


#save training dara
train_data_normal = normal_data[(normal_data['utc_time']<'2020-01-01 00:00:00')]
train_data_null = data_null[(data_null['utc_time']<'2020-01-01 00:00:00')]
train_mask = random_null_mask[(random_null_mask['utc_time']<'2020-01-01 00:00:00')]
train_org_nullmask = orign_null_mask[(orign_null_mask['utc_time']<'2020-01-01 00:00:00')]


train_data_normal.set_index(['utc_time'], inplace=True)
train_data_null.set_index(['utc_time'], inplace=True)
train_mask.set_index(['utc_time'], inplace=True)
train_org_nullmask.set_index(['utc_time'], inplace=True)

if not os.path.isdir('data/train/'):
  os.mkdir('data/train/')
train_data_normal.to_csv('data/train/train_normal.csv',index=0)
train_data_null.to_csv('data/train/train_null.csv',index=0)
train_mask.to_csv('data/train/train_mask.csv',index=0)
train_org_nullmask.to_csv('data/train/train_org_mask.csv',index=0)

