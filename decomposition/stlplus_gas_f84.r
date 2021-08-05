# install.packages("stlplus")
library(stlplus)
library(xts)
library(ggplot2)


train_data_statiton <- read.csv(file = 'data/train/train_null.csv')
test_data_statiton <- read.csv(file = 'data/test/test_null.csv')

vals =c(1:5)

for(i in vals){
  time_serise = ts(train_data_statiton[,i], frequency = 84)
  print(i)
  stl2 = stlplus(time_serise, s.window="periodic")
  trend = stl2$data[,"trend"]
  seasonal = stl2$data[,"seasonal"]
  remainder = stl2$data[,"remainder"]

  if(i==1){
    train_trend_dataframe = data.frame(col1=trend)
    train_seasonal_dataframe = data.frame(col1=seasonal)
    train_remainder_dataframe = data.frame(col1=remainder)
  } else {
    train_trend_dataframe[,paste(i)] = trend
    train_seasonal_dataframe[,paste(i)] = seasonal
    train_remainder_dataframe[,paste(i)] = remainder
  }
}

write.csv(train_trend_dataframe,paste("data/train/stlplus_train_trend84.csv", sep = ""))
write.csv(train_seasonal_dataframe,paste("data/train/stlplus_train_seasonal84.csv", sep = ""))
write.csv(train_remainder_dataframe,paste("data/train/stlplus_train_remainder84.csv", sep = ""))


for(i in vals){
  print(i)
  time_serise = ts(test_data_statiton[,i], frequency = 84)
  stl2 = stlplus(time_serise, s.window="periodic")
  trend = stl2$data[,"trend"]
  seasonal = stl2$data[,"seasonal"]
  remainder = stl2$data[,"remainder"]

  if(i==1){
    test_trend_dataframe = data.frame(col1=trend)
    test_seasonal_dataframe = data.frame(col1=seasonal)
    test_remainder_dataframe = data.frame(col1=remainder)
  } else {
    test_trend_dataframe[,paste(i)] = trend
    test_seasonal_dataframe[,paste(i)] = seasonal
    test_remainder_dataframe[,paste(i)] = remainder
  }
}

write.csv(test_trend_dataframe,paste("data/test/stlplus_test_trend84.csv", sep = ""))
write.csv(test_seasonal_dataframe,paste("data/test/stlplus_test_seasonal84.csv", sep = ""))
write.csv(test_remainder_dataframe,paste("data/test/stlplus_test_remainder84.csv", sep = ""))

