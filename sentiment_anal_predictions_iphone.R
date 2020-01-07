##########################################################################################
##                                                                                      ##
#############   Sentiment analyses toward iPhone,                                       ##
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

load("A:/B/Ubiqum/module4/sentiment_analyses/sentiment_anal_iphone.RData")

####################      optional: load models     ##########################

C50iphone<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/C50iphone.RDS")
SVMiphone<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/SVMiphone.RDS")
RFiphone<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/RFiphone.RDS")
kknniphone<-readRDS("A:/B/Ubiqum/module4/sentiment_analyses/kknniphone.RDS")

####################          import data         ############################

Predictioniphone<-read.csv('A:/B/Ubiqum/module4/sentiment_analyses/largematrixiphone.csv',
                           header=TRUE, sep=";",fill=TRUE)

####################          preprocessing       ############################

# filter out irrelevant WAPs and unneseceraily features

PredictioniphoneR<-Predictioniphone[,-nzv]

####################          predictions         ############################


Predict_C50iphone<-predict(C50iphone,PredictioniphoneR)
Predict_SVMiphone<-predict(SVMiphone,PredictioniphoneR)
Predict_RFiphone<-predict(RFiphone,PredictioniphoneR)
Predict_kknniphone<-predict(kknniphone,PredictioniphoneR)

####################            results           ############################  

summary(Predict_C50iphone)
summary(Predict_SVMiphone)
summary(Predict_RFiphone)
summary(Predict_kknniphone)

####################        end of script       ##############################