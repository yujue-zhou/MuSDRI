import pandas as pd
import numpy as np
import random
import time
from fancyimpute import KNN



test_data_statiton = pd.read_csv('data/test/test_normal.csv')
test_data_null = pd.read_csv('data/test/test_null.csv')
test_mask = pd.read_csv('data/test/test_mask.csv')


trend_matrix = pd.read_csv('data/test/mstlplus_trend.csv', index_col=0)
seasonal12_matrix = pd.read_csv('data/test/mstlplus_seasonal12.csv', index_col=0)
seasonal84_matrix = pd.read_csv('data/test/mstlplus_seasonal84.csv', index_col=0)
remainder_matrix = pd.read_csv('data/test/mstlplus_remainder.csv', index_col=0)


validate_null_number =test_data_null.isna().sum().sum()-test_data_statiton.isna().sum().sum()


k_number = 10
remainder_knn = pd.DataFrame(KNN(k=k_number).fit_transform(remainder_matrix)) 

data_mstlplus_knn = remainder_knn.to_numpy()+trend_matrix+seasonal12_matrix+seasonal84_matrix

error_mask = (test_data_statiton.fillna(0).to_numpy()-data_mstlplus_knn)*(1-test_mask).to_numpy()

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

data_mstlplus_knn = pd.DataFrame(data_mstlplus_knn)
data_mstlplus_knn.columns = test_data_statiton.columns.tolist()
data_mstlplus_knn.to_csv('./data/test/impute/mstlplus_knn.csv',index=0)


file = open('results_MAE_MRE_MSE.txt','a')
file.write('mstlplus_knn \n')
file.write('MSE = {:.5f}   '.format(mse))
file.write('MAE = {:.5f}   '.format(mae))
file.write('MRE = {:.5f}   '.format(mre))
file.write('\n')
file.close



