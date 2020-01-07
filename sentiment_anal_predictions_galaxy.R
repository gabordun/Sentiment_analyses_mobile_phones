##########################################################################################
##                                                                                      ##
#############   Sentiment analyses toward Samsung Galaxy                                ##
##                                            predictions                         ########
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
library(caret)
library(C50)
library(e1071)
library(gbm)
library(randomForest)
library(kknn)
library(mlbench)
library(export)

#################### optional: load saved environment ########################

load("A:/B/Ubiqum/module4/sentiment_analyses/sentiment_anal_galaxy.RData")

####################      optional: load models     ##########################

C50galaxy<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/C50galaxy.RDS")
SVMgalaxy<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/SVMgalaxy.RDS")
RFgalaxy<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/RFgalaxy.RDS")
kknngalaxy<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/kknngalaxy.RDS")

####################          import data         ############################

Predictiongalaxy<-read.csv('A:/B/Ubiqum/module4/sentiment_analyses/largematrixgalaxy.csv',
                           header=TRUE, sep=";",fill=TRUE)

####################          preprocessing       ############################

# filter out irrelevant WAPs and unneseceraily features

PredictiongalaxyR<-Predictiongalaxy[,-nzv]

####################          predictions         ############################


Predict_C50galaxy<-predict(C50galaxy,PredictiongalaxyR)
Predict_SVMgalaxy<-predict(SVMgalaxy,PredictiongalaxyR)
Predict_RFgalaxy<-predict(RFgalaxy,PredictiongalaxyR)
Predict_kknngalaxy<-predict(kknngalaxy,PredictiongalaxyR)

####################            results           ############################  

summary(Predict_C50galaxy)
summary(Predict_SVMgalaxy)
summary(Predict_RFgalaxy)
summary(Predict_kknngalaxy)

####################        end of script       ##############################