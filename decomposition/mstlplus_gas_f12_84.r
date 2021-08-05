# First replace the 'forecast' in the R package with the 'forecast' in the compressed folder to use MSTL+

library(forecast)


train_data_statiton <- read.csv(file = 'data/train/train_null.csv')
test_data_statiton <- read.csv(file = 'data/test/test_null.csv')

vals =c(1:5)

for(i in vals){
  print(i)
  time_serise = msts(train_data_statiton[,i], seasonal.periods = c(12, 84))

  mstlplus = mstl(time_serise)
  trend = mstlplus[, "Trend"]
  seasonal12 = mstlplus[, "Seasonal12"]
  seasonal84 = mstlplus[, "Seasonal84"]
  remainder = mstlplus[, "Remainder"]

  if(i==1){
    train_trend_dataframe = data.frame(col1=trend)
    train_seasonal12_dataframe = data.frame(col1=seasonal12)
    train_seasonal84_dataframe = data.frame(col1=seasonal84)
    train_remainder_dataframe = data.frame(col1=remainder)
  } else {
    train_trend_dataframe[,paste(i)] = trend
    train_seasonal12_dataframe[,paste(i)] = seasonal12
    train_seasonal84_dataframe[,paste(i)] = seasonal84
    train_remainder_dataframe[,paste(i)] = remainder
  }

}

write.csv(train_trend_dataframe,paste("data/train/mstlplus_trend.csv", sep = ""))
write.csv(train_seasonal12_dataframe,paste("data/train/mstlplus_seasonal12.csv", sep = ""))
write.csv(train_seasonal84_dataframe,paste("data/train/mstlplus_seasonal84.csv", sep = ""))
write.csv(train_remainder_dataframe,paste("data/train/mstlplus_remainder.csv", sep = ""))



for(i in vals){
  print(i)
  time_serise = msts(test_data_statiton[,i], seasonal.periods = c(12, 84))
  mstlplus = mstl(time_serise)
  trend = mstlplus[, "Trend"]
  seasonal12 = mstlplus[, "Seasonal12"]
  seasonal84 = mstlplus[, "Seasonal84"]
  remainder = mstlplus[, "Remainder"]

  if(i==1){
    test_trend_dataframe = data.frame(col1=trend)
    test_seasonal12_dataframe = data.frame(col1=seasonal12)
    test_seasonal84_dataframe = data.frame(col1=seasonal84)
    test_remainder_dataframe = data.frame(col1=remainder)
  } else {
    test_trend_dataframe[,paste(i)] = trend
    test_seasonal12_dataframe[,paste(i)] = seasonal12
    test_seasonal84_dataframe[,paste(i)] = seasonal84
    test_remainder_dataframe[,paste(i)] = remainder
  }

}


write.csv(test_trend_dataframe,paste("data/test/mstlplus_trend.csv", sep = ""))
write.csv(test_seasonal12_dataframe,paste("data/test/mstlplus_seasonal12.csv", sep = ""))
write.csv(test_seasonal84_dataframe,paste("data/test/mstlplus_seasonal84.csv", sep = ""))
write.csv(test_remainder_dataframe,paste("data/test/mstlplus_remainder.csv", sep = ""))
