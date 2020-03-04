# library packages 
if(!require("xgboost")) { install.packages("xgboost"); require("xgboost") }
library(xgboost)

# Read in the data and create the training and validations samples.
setwd('C:/Users/14702/Downloads')

TrainData <- read.table("newTrainingDataset.csv",sep=",",header=T,
                       stringsAsFactors=F)
ValData <- read.table("newValidationDataset.csv",sep=",",header=T,
                        stringsAsFactors=F)

#since logistic regression can only train numeric data, we transfer all catogorical variables into numerical type
# Train data 
subtr<-lapply(TrainData[,(2:22)], as.factor)
subtr<-data.frame(lapply(subtr, as.numeric))
TrainData<-cbind(TrainData$click,subtr)

# Valodation data 
subval<-lapply(ValData[,(2:22)], as.factor)
subval<-data.frame(lapply(subval, as.numeric))
ValData<-cbind(ValData$click,subval)

# Write out the data for the neural net

write.table(TrainData,file="TrainDataForBCExample.csv",sep=",",row.names=F)
write.table(ValData,file="ValDataForBCExample.csv",sep=",",row.names=F)

# Read Neural Net Output --------------------------------------------------

tmp1 <- read.table("TrYHatFromBCNN.csv",header=T,sep=",")
names(tmp1)
plot(tmp1$YHatTr,tmp1$YHatTrSM)
cor(tmp1$YHatTr,tmp1$YHatTrSM)

tmp2 <- read.table("ValYHatFromBCNN.csv",header=T,sep=",")
names(tmp2)
plot(tmp2$YHatVal,tmp2$YHatValSM)
cor(tmp2$YHatVal,tmp2$YHatValSM)

tmp2 <- read.table("ValYHatFromBCNNsm.csv",header=T,sep=",")

LL <- function(Pred,YVal){
  ll <- -mean(YVal*log(Pred)+(1-YVal)*log(1-Pred))
  return(ll)
}

tmp2<-as.numeric(tmp2)

LL(tmp2$YHatTr,ValData$V2)
X<- -(1-ValData$`ValData$click`)*log(1-tmp2$YHatVal)-ValData$`ValData$click`*log(tmp2$YHatVal)
X<- -(1-ValData$V2)*log(1-tmp2$YHatVal)-ValData$V2*log(tmp2$YHatVal)
X<- -(1-ValData$`ValData$V2`)*log(1-tmp2$YHatVal)-ValData$`ValData$V2`*log(tmp2$YHatVal)
X<- -(1-ValData$`ValData$V2`)*log(1-tmp2$YHatValSM)-ValData$`ValData$V2`*log(tmp2$YHatValSM)
mean(X)

write.table(X,file="X.csv",sep=",",row.names=F)


# --------------------------------------------------



# Xgboost
xgb <- xgboost(data = data.matrix(TrainData[,-1]), label = TrainData$`TrainData$click`, 
               max.depth = 4,  min_child_weight = 5,
               nround=1000, objective = "binary:logistic")

# xgb <- xgboost(data = data.matrix(TrainData[,-c(1:3)]), label = TrainData$V2,nround=50, objective = "binary:logistic")

pred <- predict(xgb, data.matrix(ValData[,-1]))
# pred <- predict(xgb, data.matrix(ValData[,-c(1:3)]))

LL(pred,ValData$`ValData$click`)
# LL(pred,ValData$V2)


# labels <- as.vector(unique(TrainData$click))


###########end




TrainData <- read.table("Train-001-Random-training.csv",sep=",",header=F,stringsAsFactors=F)
ValData <- read.table("Train-001-Random-validation.csv",sep=",",header=F,stringsAsFactors=F)

# load(file="SpamdataPermutation.RData")
# DataOrig <- DataOrig[ord,]

# Doing a 60-40 split
# TrainInd <- ceiling(nrow(DataOrig)*0.6)
# TrainData <- DataOrig[1:TrainInd,]
# ValData <- DataOrig[(TrainInd+1):nrow(DataOrig),]

# TrainData$V3<-substr(TrainData$V3, 7, 8)
# ValData$V3<-substr(ValData$V3, 7, 8)


subval<-ValData[,-(12:13)]
subval<-lapply(subval[,(3:22)], as.factor)
subval<-data.frame(lapply(subval, as.numeric))
ValData<-cbind(ValData$V2,subval)


subval<-lapply(ValData[,6:14], as.factor)
subval<-data.frame(lapply(subval, as.numeric))

ValData<-ValData[,-(6:14)]
ValData<-cbind(ValData,subval)

TrainData[] <- lapply(TrainData, as.factor)
TrainData[] <- lapply(TrainData, as.numeric)
ValData[] <- lapply(ValData, as.factor)
ValData[] <- lapply(ValData, as.numeric)




colnames(TrainData) <- c("id","click","hour","C1","banner_pos","site_id","side_domain","site_category","app_id","app_domain","app_category",
                         "device_id","device_ip","device_model","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21")