
library(ggplot2)
library(forecast)


train_data_statiton <- read.csv(file = 'data/train/train_null.csv')
test_data_statiton <- read.csv(file = 'data/test/test_null.csv')

seasonality_period_2 = 12
seasonality_period_3 = 84


vals =c(1:5)

for(i in vals){

  regessor2 = fourier(msts(train_data_statiton[,i], seasonal.periods = seasonality_period_2), K=c(5))
  regessor3 = fourier(msts(train_data_statiton[,i], seasonal.periods = seasonality_period_3), K=c(20))

  seasonality2 <- rowSums(regessor2)
  seasonality3 <- rowSums(regessor3)


  if(i==1){
    train_seasonality2_dataframe = data.frame(col1=seasonality2)
    train_seasonality3_dataframe = data.frame(col1=seasonality3)

  } else {
    train_seasonality2_dataframe[,paste(i)] = seasonality2
    train_seasonality3_dataframe[,paste(i)] = seasonality3

  }

}

write.csv(train_seasonality2_dataframe,paste("data/train/fourier_seasonality1.csv", sep = ""))
write.csv(train_seasonality3_dataframe,paste("data/train/fourier_seasonality2.csv", sep = ""))




for(i in vals){

  regessor2 = fourier(msts(test_data_statiton[,i], seasonal.periods = seasonality_period_2), K=c(5))
  regessor3 = fourier(msts(test_data_statiton[,i], seasonal.periods = seasonality_period_3), K=c(20))

  seasonality2 <- rowSums(regessor2)
  seasonality3 <- rowSums(regessor3)


  if(i==1){
    test_seasonality2_dataframe = data.frame(col1=seasonality2)
    test_seasonality3_dataframe = data.frame(col1=seasonality3)

  } else {
    test_seasonality2_dataframe[,paste(i)] = seasonality2
    test_seasonality3_dataframe[,paste(i)] = seasonality3

  }

}

write.csv(test_seasonality2_dataframe,paste("data/test/fourier_seasonality1.csv", sep = ""))
write.csv(test_seasonality3_dataframe,paste("data/test/fourier_seasonality2.csv", sep = ""))
