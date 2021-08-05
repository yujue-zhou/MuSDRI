import pandas as pd
import numpy as np
import random
import time
from rstl import STL


freq = 84

test_data_statiton = pd.read_csv('data/test/test_normal.csv')
test_data_null = pd.read_csv('data/test/test_null.csv')
test_mask = pd.read_csv('data/test/test_mask.csv')


trend_matrix = pd.read_csv('data/test/stlplus_test_trend'+str(freq)+'.csv', index_col=0)
seasonal_matrix = pd.read_csv('data/test/stlplus_test_seasonal'+str(freq)+'.csv', index_col=0)
remainder_matrix = pd.read_csv('data/test/stlplus_test_remainder'+str(freq)+'.csv', index_col=0)


validate_null_number =test_data_null.isna().sum().sum()-test_data_statiton.isna().sum().sum()


remainder_last = remainder_matrix.fillna(method='pad') 
remainder_last = remainder_last.fillna(method='backfill') 

data_stlplus_last = remainder_last.to_numpy()+trend_matrix+seasonal_matrix


error_mask = (test_data_statiton.fillna(0).to_numpy()-data_stlplus_last)*(1-test_mask).to_numpy()

mse_error = error_mask**2
mae_error = mre_error = abs(error_mask)

total_error_MSE = mse_error.sum().sum()
total_error_MAE = mae_error.sum().sum()
total_error_MRE = mre_error.sum().sum()

total_label_MRE = abs(test_data_statiton.fillna(0).to_numpy()*(1-test_mask).to_numpy()).sum().sum()

mse = total_error_MSE/validate_null_number
mae = total_error_MAE/validate_null_number
mre = total_error_MRE/total_label_MRE


print(mse)
print(mae)
print(mre)

data_stlplus_last = pd.DataFrame(data_stlplus_last)
data_stlplus_last.columns = test_data_statiton.columns.tolist()
data_stlplus_last.to_csv('./data/test/impute/stlplus_last'+str(freq)+'.csv',index=0)


file = open('results_MAE_MRE_MSE.txt','a')
file.write('stlplus_last \n')
file.write('frequence = {},   '.format(freq))
file.write('\n')
file.write('MSE = {:.5f}   '.format(mse))
file.write('MAE = {:.5f}   '.format(mae))
file.write('MRE = {:.5f}   '.format(mre))
file.write('\n')
file.close


