setwd('D:\\Sem2\\Data Mining\\Project')

library('DMwR')
library(rpart)
library(randomForest)
library(ROCR)



######################### Start of data cleansing ############################


pMiss = function(x) {
  return(sum(is.na(x))*100/length(x))
}

removeMissingColumns = function(x, tol=0.02){
  if ( (sum(is.na(x))/length(x)) >tol){
      return(FALSE)
  }
  else{
      return(TRUE)
  }
}

df_train = read.csv('train.csv')
df_test = read.csv('test.csv')

#Working with samples of train and test (10,000 rows each) to reduce computation during development cycle

df_train = read.csv('train_sample.csv')
df_test = read.csv('test_sample.csv')

print(paste0("Number of rows in train with atleast one missing value: ", sum(!complete.cases(df_train)), " out of ", nrow(df_train) ))

df_train_clean = df_train[(complete.cases(df_train)),] #Removing all rows that has atleast one missing column. Segment 1 of data set.
df_train_missing = df_train[(!complete.cases(df_train)),] #All the rest of the rows. Segment 2 of data set.
df_train_missing_clean = df_train_missing[apply(df_train_missing, 2, removeMissingColumns)] #Cleaning up segment 2 by removing columns
df_train_missing_clean = na.omit(df_train_missing_clean) #Cleaning up segment 2 by removing the few rows that still have a missing value


z = cor(df_train_clean$target, as.matrix(df_train_clean[sapply(df_train_clean, is.numeric)]))
levelplot(z)

z1 = cor(df_train_missing_clean$target, as.matrix(df_train_missing_clean[sapply(df_train_missing_clean, is.numeric)]))
levelplot(z1)

########################### End of data cleansing ######################

######################### Start of modeling ############################

#model <-glm(formula= target~.,data=df_train_missing_clean, family = binomial)

cartmodel = rpart(formula= as.factor(target) ~.,data=df_train_missing_clean, method = 'class')
plot(cartmodel)
text(cartmodel)
plotcp(cartmodel)

rf_clean = randomForest(as.factor(target) ~ ., data = df_train_clean[sapply(df_train_clean, is.numeric)], ntree = 200)

pred = predict(rf_clean)

table(df_train_clean$target,pred,dnn=c('True','Predicted'))


rf_missing = randomForest(as.factor(target) ~ ., data = df_train_missing_clean[sapply(df_train_missing_clean, is.numeric)], ntree = 200)

pred = predict(rf_missing)

table(df_train_missing_clean$target,pred,dnn=c('True','Predicted'))







