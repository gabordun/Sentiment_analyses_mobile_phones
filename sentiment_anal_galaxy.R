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

load("A:/B/Ubiqum/module4/sentiment_analyses/sentiment_anal_galaxy.RData")

#################### load database (from a csv file)  ################################

iphone<-read.table('A:/B/Ubiqum/module4/sentiment_analyses/iphone_smallmatrix_labeled_8d.csv',
                      header=TRUE, sep=",",fill=TRUE)
galaxy<-read.table('A:/B/Ubiqum/module4/sentiment_analyses/galaxy_smallmatrix_labeled_8d.csv',
                   header=TRUE, sep=",",fill=TRUE)

#################### Part 1: preprocessing, feature selection #################################

#checking NA values
NAcolumns<-colnames(galaxy)[colSums(is.na(galaxy)) > 0]

#############   investigating, preprocessing dataset - correlations ####################

##computing correlations
Corrgalaxy<-cor(galaxy)

# matrix of the p-value of the correlation
p.mat <- cor.mtest(galaxy)

#plotting correlations
corrplot(Corrgalaxy, method="number", type="lower", addCoef.col = TRUE,
         addCoefasPercent = TRUE,
         tl.srt=45, #p.mat=p.mat,
         sig.level=0.05, insig="blank", diag=FALSE)

#checking significant correlations with galaxysentiment as depending variable

thresholdgalaxy<-data.frame(Corrgalaxy)
thresholdgalaxysenti<-data.frame(thresholdgalaxy$galaxysentiment)
thresholdgalaxysenti<-cbind(thresholdgalaxysenti,colnames(galaxy))
thresholdgalaxysenti<-thresholdgalaxysenti[-59,]

#plot the result
corrgalaxyplot<-ggplot(thresholdgalaxysenti, 
                       aes(x=thresholdgalaxysenti$colnames(galaxy), 
                           y=thresholdgalaxysenti$thresholdgalaxy.galaxysentiment))+
  geom_bar(stat = "identity")+theme(axis.text.x = element_text(angle=90, hjust=1))

corrgalaxyplot

##check near-zero variance

nzvMetrics <- nearZeroVar(galaxy, saveMetrics = TRUE)
nzvMetrics

nzv <- nearZeroVar(galaxy, saveMetrics = FALSE) 
nzv

# create a new data set and remove near zero variance features
galaxyNZV <- galaxy[,-nzv]
str(galaxyNZV)

## recursive feature elimination
# let's sample the data before using RFE
set.seed(123)
galaxySample <- galaxy[sample(1:nrow(galaxy), 100, replace=FALSE),]

# set up rfeControl with randomforest, repeated cross validation and no updates
ctrl <- rfeControl(functions = rfFuncs, 
                   method = "repeatedcv",
                   repeats = 3,
                   verbose = FALSE)

# use rfe and omit the response variable (attribute 59 iphonesentiment) 
rfeResults <- rfe(galaxySample[,1:58], 
                  galaxySample$galaxysentiment, 
                  sizes=(1:58), 
                  rfeControl=ctrl)

# get results
rfeResults

# Plot results
plot(rfeResults, type=c("g", "o"))

#################       Part2:   modelling, training, full data set         #######################

set.seed(123)

#turn depending varioable into a factor
galaxy$galaxysentiment<-as.factor(galaxy$galaxysentiment)

#partioning to training and testing sets
inTraining <- createDataPartition(galaxy$galaxysentiment, p = 0.7, list = FALSE)
training <- galaxy[inTraining, ]
testing <- galaxy[-inTraining, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 3)

#C5.0
Mod1C50<-C5.0(galaxysentiment~.,data=training, trials = 1)
Mod1C50
varImp(Mod1C50)

#SVM, linear kernel
Mod2SVM<-train(galaxysentiment~.,data=training, method="svmLinear",
               scaled=c(),trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneLength=2)
Mod2SVM
varImp(Mod2SVM)

#Random forest
Mod3Rf<-randomForest(galaxysentiment~.,data=training, mtry=40,ntree=120,
                     trControl=fitControl)
Mod3Rf

#kkNN
Mod4kknn <- train.kknn(galaxysentiment~ .,training,
                       kmax = 10,
                       ks = NULL,
                       distance = 2)
cv.kknn(galaxysentiment~., testing, kcv = 10)
Mod4kknn

#############   model validations ##################

TestC50<-predict(Mod1C50,testing)
TestSVM<-predict(Mod2SVM,testing)
TestRF<-predict(Mod3Rf,testing)
Testkknn<-predict(Mod4kknn,testing)
TestMatrix<-data.frame(TestC50,TestSVM,TestRF,Testkknn,testing$galaxysentiment)

cmC50 <- confusionMatrix(TestC50, testing$galaxysentiment)
cmSVM <- confusionMatrix(TestSVM, testing$galaxysentiment)
cmRF <- confusionMatrix(TestRF, testing$galaxysentiment)
cmkknn <- confusionMatrix(Testkknn, testing$galaxysentiment)

cmC50
cmSVM
cmRF
cmkknn

#################      Part3: modelling, training, reduced data set         #######################

set.seed(123)

#turn depending varioable into a factor
galaxyNZV$galaxysentiment<-as.factor(galaxyNZV$galaxysentiment)

#partioning to training and testing sets
inTrainingR <- createDataPartition(galaxyNZV$galaxysentiment, p = 0.7, list = FALSE)
trainingR <- galaxyNZV[inTrainingR, ]
testingR <- galaxyNZV[-inTrainingR, ]
fitControl<-trainControl(method="repeatedcv", number=10, repeats = 3)

#C5.0
ModRC50<-C5.0(galaxysentiment~.,data=trainingR, trials = 1)
ModRC50
varImp(ModRC50)

#SVM, linear kernel
ModRSVM<-train(galaxysentiment~.,data=trainingR, method="svmLinear",
               scaled=c(),trControl=fitControl,
               preProcess = c("center", "scale"),
               tuneLength=2)
ModRSVM
varImp(ModRSVM)

#Random forest
ModRRf<-randomForest(galaxysentiment~.,data=trainingR, mtry=12,ntree=120,
                     trControl=fitControl)
ModRRf

#kkNN
ModRkknn <- train.kknn(galaxysentiment~ .,trainingR,
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
                        testingR$galaxysentiment)

cmC50R <- confusionMatrix(TestRC50, testingR$galaxysentiment)
cmSVMR <- confusionMatrix(TestRSVM, testingR$galaxysentiment)
cmRFR <- confusionMatrix(TestRRF, testingR$galaxysentiment)
cmkknnR <- confusionMatrix(TestRkknn, testingR$galaxysentiment)

cmC50R
cmSVMR
cmRFR
cmkknnR

###############      save final models     #################################

saveRDS(ModRC50, "A:/B/Ubiqum/module4/sentiment_analyses/C50galaxy.RDS")
saveRDS(ModRSVM, "A:/B/Ubiqum/module4/sentiment_analyses/SVMgalaxy.RDS")
saveRDS(ModRRf, "A:/B/Ubiqum/module4/sentiment_analyses/RFgalaxy.RDS")
saveRDS(ModRkknn, "A:/B/Ubiqum/module4/sentiment_analyses/kknngalaxy.RDS")

###############      delete plots         ##################################

graphics.off()

################     end of script         #################################
