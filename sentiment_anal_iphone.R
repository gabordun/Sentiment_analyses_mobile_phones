##########################################################################################
##                                                                                      ##
#############   Sentiment analysis toward Samsung Galaxy                ##################
##              Author: Gabor Dunai                                                     ##
##              Version: 1.0                                                            ##
##              Date: 01.2020                                                           ##
##                                                                                      ##
##########################################################################################

####################  Part 0: directory, libraries, dataset   ################

#################### set directory, call libraries ###########################

setwd("A:/B/Ubiqum/module4/sentiment_analyses")
library(dplyr)
library(ggplot2)
library(plotly)
library(corrplot)
library(caret)
library(C50)
library(e1071)
library(randomForest)
library(kknn)
library(mlbench)
library(export)

#################### load basic environment (from a file already saved) ####################

load("A:/B/Ubiqum/module4/sentiment_analyses/sentiment_anal_iphone.RData")

#################### load database (from a csv file)  ################################

iphone<-read.table('A:/B/Ubiqum/module4/sentiment_analyses/iphone_smallmatrix_labeled_8d.csv',
                      header=TRUE, sep=",",fill=TRUE)
galaxy<-read.table('A:/B/Ubiqum/module4/sentiment_analyses/galaxy_smallmatrix_labeled_8d.csv',
                   header=TRUE, sep=",",fill=TRUE)

#################### Part 1: preprocessing, feature selection #################################

#checking NA values
NAcolumns<-colnames(iphone)[colSums(is.na(iphone)) > 0]

#############   investigating, preprocessing dataset - correlations ####################

##computing correlations
Corriphone<-cor(iphone)

# matrix of the p-value of the correlation
p.mat <- cor.mtest(iphone)

#plotting correlations
corrplot(Corriphone, method="number", type="lower", addCoef.col = TRUE,
         addCoefasPercent = TRUE,
         tl.srt=45, #p.mat=p.mat,
         sig.level=0.05, insig="blank", diag=FALSE)

#checking significant correlations with iphonesentiment as depending variable

thresholdiphone<-data.frame(Corriphone)
thresholdiphonesenti<-data.frame(thresholdiphone$iphonesentiment)
thresholdiphonesenti<-cbind(thresholdiphonesenti,colnames(iphone))
thresholdiphonesenti<-thresholdiphonesenti[-59,]

#plot the result
Corriphoneplot<-ggplot(thresholdiphonesenti, 
                       aes(x=thresholdiphonesenti$'colnames(iphone)', 
                           y=thresholdiphonesenti$iphonesentiment))+
  geom_bar(stat = "identity")+theme(axis.text.x = element_text(angle=90, hjust=1))

Corriphoneplot

##check near-zero variance

nzvMetrics <- nearZeroVar(iphone, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(iphone, saveMetrics = FALSE) 
nzv

# create a new data set and remove near zero variance features
iphoneNZV <- iphone[,-nzv]
str(iphoneNZV)

## recursive feature elimination
# let's sample the data before using RFE
set.seed(123)
iphoneSample <- iphone[sample(1:nrow(iphone), 100, replace=FALSE),]

# set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)

# use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(iphoneSample[,1:58], 
                  iphoneSample$iphonesentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

#################       Part2:   modelling, training, full data set         #######################

set.seed(123)

#turn depending varioable into a factor
iphone$iphonesentiment<-as.factor(iphone$iphonesentiment)

#partioning to training and testing sets
inTraining <- createDataPartition(iphone$iphonesentiment, p = 0.7, list = FALSE)
training <- iphone[inTraining, ]
testing <- iphone[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 3)

#C5.0
Mod1C50<-C5.0(iphonesentiment~.,data=training, trials = 1)
Mod1C50
varImp(Mod1C50)

#SVM, linear kernel
Mod2SVM<-train(iphonesentiment~.,data=training, method="svmLinear",
               scaled=c(),trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneLength=2)
Mod2SVM
varImp(Mod2SVM)

#Random forest
Mod3Rf<-randomForest(iphonesentiment~.,data=training, mtry=40,ntree=120,
                     trControl=fitControl)
Mod3Rf

#kkNN
Mod4kknn <- train.kknn(iphonesentiment~ .,training,
                       kmax = 10,
                       ks = NULL,
                       distance = 2)
cv.kknn(iphonesentiment~., testing, kcv = 10)
Mod4kknn

#############   model validations ##################

TestC50<-predict(Mod1C50,testing)
TestSVM<-predict(Mod2SVM,testing)
TestRF<-predict(Mod3Rf,testing)
Testkknn<-predict(Mod4kknn,testing)
TestMatrix<-data.frame(TestC50,TestSVM,TestRF,Testkknn,testing$iphonesentiment)

cmC50 <- confusionMatrix(TestC50, testing$iphonesentiment)
cmSVM <- confusionMatrix(TestSVM, testing$iphonesentiment)
cmRF <- confusionMatrix(TestRF, testing$iphonesentiment)
cmkknn <- confusionMatrix(Testkknn, testing$iphonesentiment)

cmC50
cmSVM
cmRF
cmkknn

#################      Part3: modelling, training, reduced data set         #######################

set.seed(123)

#turn depending varioable into a factor
iphoneNZV$iphonesentiment<-as.factor(iphoneNZV$iphonesentiment)

#partioning to training and testing sets
inTrainingR <- createDataPartition(iphoneNZV$iphonesentiment, p = 0.7, list = FALSE)
trainingR <- iphoneNZV[inTrainingR, ]
testingR <- iphoneNZV[-inTrainingR, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 3)

#C5.0
ModRC50<-C5.0(iphonesentiment~.,data=trainingR, trials = 1)
ModRC50
varImp(ModRC50)

#SVM, linear kernel
ModRSVM<-train(iphonesentiment~.,data=trainingR, method="svmLinear",
               scaled=c(),trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneLength=2)
ModRSVM
varImp(ModRSVM)

#Random forest
ModRRf<-randomForest(iphonesentiment~.,data=trainingR, mtry=12,ntree=120,
                     trControl=fitControl)
ModRRf

#kkNN
ModRkknn <- train.kknn(iphonesentiment~ .,trainingR,
                       kmax = 20,
                       ks = NULL,
                       distance = 2)
cv.kknn(iphonesentiment~., testingR, kcv = 10)
ModRkknn

#############   model validations ##################

TestRC50<-predict(ModRC50,testingR)
TestRSVM<-predict(ModRSVM,testingR)
TestRRF<-predict(ModRRf,testingR)
TestRkknn<-predict(ModRkknn,testingR)
TestMatrixR<-data.frame(TestRC50,TestRSVM,TestRRF,TestRkknn,
                        testingR$iphonesentiment)

cmC50R <- confusionMatrix(TestRC50, testingR$iphonesentiment)
cmSVMR <- confusionMatrix(TestRSVM, testingR$iphonesentiment)
cmRFR <- confusionMatrix(TestRRF, testingR$iphonesentiment)
cmkknnR <- confusionMatrix(TestRkknn, testingR$iphonesentiment)

cmC50R
cmSVMR
cmRFR
cmkknnR

###############      save final models     #################################

saveRDS(ModRC50, "A:/B/Ubiqum/module4/sentiment_analyses/C50iphone.RDS")
saveRDS(ModRSVM, "A:/B/Ubiqum/module4/sentiment_analyses/SVMiphone.RDS")
saveRDS(ModRRf, "A:/B/Ubiqum/module4/sentiment_analyses/RFiphone.RDS")
saveRDS(ModRkknn, "A:/B/Ubiqum/module4/sentiment_analyses/kknniphone.RDS")

###############      delete plots         ##################################

graphics.off()

################     end of script         #################################
