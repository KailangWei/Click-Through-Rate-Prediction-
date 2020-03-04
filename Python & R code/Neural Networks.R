# Read in the data and create the training, validations and test samples.
setwd('C:/Users/14702/Downloads')

TrainData <- read.table("newTrainingDataset.csv",sep=",",header=T,stringsAsFactors=F)
ValData <- read.table("newValidationDataset.csv",sep=",",header=T,stringsAsFactors=F)
TestData <- read.table("ProcessedTestData-Robert.csv",sep=",",header=T,stringsAsFactors=F)
ValData2 <- read.table("newValidationDataset2.csv",sep=",",header=T,stringsAsFactors=F)

#since Neural Networks can only train numeric data, we transfer all catogorical variables into numerical type
# Train data 
subtr<-lapply(TrainData[,(2:22)], as.factor)
subtr<-data.frame(lapply(subtr, as.numeric))
TrainData<-cbind(TrainData$click,subtr)

# Valodation data 
subval<-lapply(ValData[,(2:22)], as.factor)
subval<-data.frame(lapply(subval, as.numeric))
ValData<-cbind(ValData$click,subval)

# Valodation data 2
subval2<-lapply(ValData2[,(2:22)], as.factor)
subval2<-data.frame(lapply(subval2, as.numeric))
ValData2<-cbind(ValData2$click,subval2)

# Test data 
TestData<-lapply(TestData, as.factor)
TestData<-data.frame(lapply(TestData, as.numeric))

# Write out the data for the neural net
write.table(TrainData,file="TrainDataForBCExample.csv",sep=",",row.names=F)
write.table(ValData,file="ValDataForBCExample.csv",sep=",",row.names=F)
write.table(TestData,file="TestDataForBCExample.csv",sep=",",row.names=F)


# Read Neural Net Output --------------------------------------------------
tmp1 <- read.table("TrYHatFromBCNN.csv",header=T,sep=",")
tmp2 <- read.table("ValYHatFromBCNN.csv",header=T,sep=",")

# compare three nn models
ll4<- -(1-ValData$`ValData$click`)*log(1-tmp2$YHatVal4)-ValData$`ValData$click`*log(tmp2$YHatVal4)
mean(ll4) # log loss = 0.421
ll10<- -(1-ValData$`ValData$click`)*log(1-tmp2$YHatVal10)-ValData$`ValData$click`*log(tmp2$YHatVal10)
mean(ll10) # log loss = 0.456
llsm<- -(1-ValData$`ValData$click`)*log(1-tmp2$YHatValSM)-ValData$`ValData$click`*log(tmp2$YHatValSM)
mean(llsm) # log loss = 0.432

# it turns out Neural Net Model has optimal outcomes with 4 layes of 4 neurons probability output

# calculate log loss on validation data 2
tmp3 <- read.table("ValYHatFromBCNN2.csv",header=T,sep=",")
llv2<--(1-ValData$`ValData$click`)*log(1-tmp3$YHatVal)-ValData$`ValData$click`*log(tmp3$YHatVal)
mean(llv2) # log loss = 0.4215
