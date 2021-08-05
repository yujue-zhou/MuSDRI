import pandas as pd
import numpy as np
import random
import time
import os

if not os.path.exists('./data/test/impute'):
  os.makedirs('./data/test/impute') 


test_data_statiton = pd.read_csv('data/test/test_normal.csv')
test_data_null = pd.read_csv('data/test/test_null.csv')
test_mask = pd.read_csv('data/test/test_mask.csv')

start_time = time.time()


validate_null_number =test_data_null.isna().sum().sum()-test_data_statiton.isna().sum().sum()


# last 补全缺失值
data_last = test_data_null.fillna(method='pad') 
data_last = data_last.fillna(method='backfill') 

error_mask = (test_data_statiton.fillna(0).to_numpy()-data_last)*(1-test_mask).to_numpy()

mse_error = error_mask**2
mae_error = mre_error = abs(error_mask)

total_error_MSE = mse_error.sum().sum()
total_error_MAE = mae_error.sum().sum()
total_error_MRE = mre_error.sum().sum()

total_label_MRE = abs(test_data_statiton.fillna(0).to_numpy()*(1-test_mask).to_numpy()).sum().sum()

data_last = pd.DataFrame(data_last)
data_last.columns = test_data_statiton.columns.tolist()
data_last.to_csv('./data/test/impute/last.csv',index=0)


mse = total_error_MSE/validate_null_number
mae = total_error_MAE/validate_null_number
mre = total_error_MRE/total_label_MRE


print(mse)
print(mae)
print(mre)


file = open('results_MAE_MRE_MSE.txt','a')
file.write('last \n')
file.write('MSE = {:.5f}   '.format(mse))
file.write('MAE = {:.5f}   '.format(mae))
file.write('MRE = {:.5f}   '.format(mre))
file.write('\n')
file.close
