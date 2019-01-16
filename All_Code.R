rm(list= ls()[!(ls() %in% c('ds'))])
ds = read.csv("/home/sharm364/Downloads/FraudData.csv")
d=ds
d$step = as.integer(d$step)
d$type = as.factor(d$type)
d$amount = as.numeric(d$amount)
d$nameOrig = as.character(d$nameOrig)
d$oldbalanceOrg = as.numeric(d$oldbalanceOrg)
d$newbalanceOrig = as.numeric(d$newbalanceOrig)
d$nameDest = as.character(d$nameDest)
d$oldbalanceDest = as.numeric(d$oldbalanceDest)
d$newbalanceDest = as.numeric(d$newbalanceDest)
d$isFraud = as.integer(d$isFraud)
d$isFlaggedFraud = as.factor(d$isFlaggedFraud)


library(caret)
library(pROC)
##taking the training set
dfraud = d[d$isFraud == 1,]
dNfraud = d[d$isFraud == 0,]
sample = createDataPartition(y =dNfraud$type ,   # outcome variable
                             p = .0012924,   # % of training data you want
                             list = F)
dNfraudPart <- dNfraud[sample,]  # training data set
dTrain = rbind(dfraud,dNfraudPart)

##taking test set
sample = createDataPartition(y =d$isFraud ,   # outcome variable
                             p = .001,   # % of training data you want
                             list = F)
dTest = d[sample,]
d=rbind(dTest,dTrain)
rm(dfraud,dNfraud,dNfraudPart)

library(caret)
##Feature Engineering
d['Day']=d['step']%/%24
d['TimeOfDay']=d['step']%%24
d['SourTrans']=d['oldbalanceOrg']-d['newbalanceOrig']
d['DestTrans']=d['newbalanceDest']-d['oldbalanceDest']
d['A-ST']=d['amount']-d['SourTrans']
d['A-DT']=d['DestTrans']-d['amount']
d['ST-DT']=d['SourTrans']-d['DestTrans']
d['step']  = NULL

d<-d[,c(10, 1:9, 11:ncol(d))]
X = d[(d$type == 'TRANSFER') | (d$type == 'CASH_OUT'),]
X=X[,-4]
FD<-d
str(FD)
#write.csv(FD, file = "Deepika.csv")
FD$nameOrig<-NULL
FD$nameDest<-NULL
FD$isFlaggedFraud = NULL

names(FD)[1] = 'y'
dummies <- dummyVars(y ~ ., data = FD)            # create dummyes for Xs
ex <- data.frame(predict(dummies, newdata = FD))  # actually creates the dummies
names(ex) <- gsub("\\.", "", names(ex))          # removes dots from col names
FD <- cbind(FD$y, ex)                              # combine your target variable with Xs
names(FD)[1] <- "y"                               # make target variable called 'y'
rm(dummies, ex)                                  # delete temporary things we no longer need

# calculate correlation matrix using Pearson's correlation formula
FD$type = NULL
descrCor <-  cor(FD[,2:ncol(FD)])                         # correlation matrix
highCorr <- sum(abs(descrCor[upper.tri(descrCor)]) > .85) # number of Xs having a corr > some value
summary(descrCor[upper.tri(descrCor)])                    # summarize the correlations

# which columns in your correlation matrix have a correlation greater than some
# specified absolute cutoff?
highlyCorDescr <- findCorrelation(descrCor, cutoff = 0.85)
filteredDescr <- FD[,2:ncol(FD)][,-highlyCorDescr] # remove those specific columns from your dataset
descrCor2 <- cor(filteredDescr)                  # calculate a new correlation matrix

# summarize those correlations to see if all features are now within our range
summary(descrCor2[upper.tri(descrCor2)])

# update our d dataset by removing those filtered variables that were highly correlated
FD <- cbind(FD$isFraud, filteredDescr)
names(FD)[1] <- "y"

# create a column of 1s. This will help identify all the right linear combos
FD <- cbind(rep(1, nrow(FD)), FD[,2:ncol(FD)])
names(FD)[1] <- "ones"
# identify the columns that are linear combos
comboInfo <- findLinearCombos(FD)
comboInfo

# remove columns identified that led to linear combos
FD <- FD[, -comboInfo$remove]

# remove the "ones" column in the first column
FD <- FD[, c(2:ncol(FD))]
str(FD)
# Add the target variable back to our data.frame
#FD <- cbind(y, FD)

#rm(y, comboInfo)  # clean up
################################################################################
# Standardize (and/ normalize) your input features.
################################################################################

# To make sure I do not standardize the dummy variables I'll create a set that 
# contains the 0/1 variables (dCats) and the numeric features (dNums) 
numcols <- apply(X=FD, MARGIN=2, function(c) sum(c==0 | c==1)) != nrow(FD)
catcols <- apply(X=FD, MARGIN=2, function(c) sum(c==0 | c==1)) == nrow(FD)
FDNums <- FD[,numcols]
FDCats <- FD[,catcols]
# Step 1) figures out the means, standard deviations, other parameters, etc. to 
# transform each variable
preProcValues <- preProcess(FDNums[,2:ncol(FDNums)], method = c("center","scale"))
# Here we standardize the input features (Xs) using the preProcess() function 
# by performing a min-max normalization (aka "range" in caret).
FDNums <- predict(preProcValues, FDNums)

# combine the standardized numeric features with the dummy vars
FD <- cbind(FDNums, FDCats)
str(FD)
names(FD)[dim(FD)[2]] = 'y'
FD$y = as.factor(FD$y)
FD$y = make.names(FD$y)
ctrl <- trainControl(method = "cv",
                     number = 5,classProbs = T,  # if you want probabilities
                     summaryFunction = twoClassSummary, # for classification
                     allowParallel=T)
FD$result = NULL

dim(FD[FD$y == 'X1',])
train1 = FD[1:16428,]
test1 =  FD[16429:22791,]
dim(test1[train1$y == 'X0',])
sample = createDataPartition(y = train1$y,   # outcome variable
                             p = .70,   # % of training data you want
                             list = F)
train = train1[sample,]
test = train1[-sample,]



treeModel <- train(y~.,method ='rpart',data = train,trControl = ctrl)
TreeResult<-predict(treeModel, test, type="prob")[,2]
predictions = TreeResult
actual_value = test$y
test$y <- ifelse(test$y == "X0", as.integer(0), test$y)
test$y <- ifelse(test$y == "X1", as.integer(1), test$y)


#plot(auc, ylim=c(0,1), print.thres=TRUE, main=paste('AUC:',round(auc$auc[[1]],2)))
abline(h=1,col='blue',lwd=2)
abline(h=0,col='red',lwd=2)
abline(v=1,col='red',lwd=2)
#test$result = TreeResult
#test$result = make.names(test$result)
#dim(test[test$y==test$result & test$y == 'X1',])
#dim(test[test$y == 'X1',])
#test$y = as.factor(test$y)
#test$result = as.factor(test$result)
#levels(test$y) = c('X0','X1')
#confusionMatrix(test$result,test$y)

rfModel <- train(y~.,method ='rf',data = train,trControl = ctrl)
prediction<-predict(rfModel, test)

glm <- train(y ~., method="glm", family="binomial",data = train)
summary(glm)

train$y <- ifelse(train$y == "X0", 0, train$y)
train$y <- ifelse(train$y == "X1", 1, train$y)
train$y = as.numeric(train$y)
dtrain <- xgb.DMatrix(data = as.matrix(train), label = train$y)
xgb <- xgboost(data = data.matrix(train[,1:7]), label = train$y,
               max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 0)

xgbResult<-predict(xgb, data.matrix(test[,1:7]))
prediction <- as.numeric(xgbResult > 0.5)
test$y = as.factor(test$y)
prediction = as.factor(prediction)
test$y =  make.names(test$y)
prediction = make.names(prediction)
prediction = as.factor(prediction)
test$y = as.factor(test$y)
confusionMatrix(prediction,test$y)
roc_obj <- roc(test$y, prediction)
auc(roc_obj)


xgbResult<-predict(xgb, data.matrix(train[,1:7]))
prediction <- as.numeric(xgbResult > 0.5)
train$y = as.factor(train$y)
prediction = as.factor(prediction)
train$y =  make.names(train$y)
prediction = make.names(prediction)
prediction = as.factor(prediction)
train$y = as.factor(train$y)
confusionMatrix(prediction,train$y)

rfModelResult<-predict(rfModel, train[,1:7] )
confusionMatrix(rfModelResult,train$y)
roc_obj <- roc(test$y, rfModelResult)
auc(roc_obj)

rfModelResult<-predict(rfModel, test[,1:7] )
confusionMatrix(rfModelResult,test$y)
roc_obj <- roc(test$y, rfModelResult)
auc(roc_obj)

glmModelResult<-predict(glm, train[,1:7] )
confusionMatrix(glmModelResult,train$y)
roc_obj <- roc(train$y, glmModelResult)
auc(roc_obj)

glmModelResult<-predict(glm, test[,1:7] )
confusionMatrix(glmModelResult,test$y)
roc_obj <- roc(test$y, glmModelResult)
auc(roc_obj)
