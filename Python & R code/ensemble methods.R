# ensemble methods 

setwd('C:/Users/14702/Downloads')

# read in validation data 2
ValData <- read.table("newValidationDataset2.csv",sep=",",header=T,stringsAsFactors=F)
YVal<-ValData$click

# read in predictions on validation data 2
dt<-read.table('DecisionTreePredictionsVal2.csv',header=T)
lg<-read.table('LRValidation2Predictions (1).csv',header=T)
nn<-read.table('ValYHatFromBCNN2.csv',header=T)
xgb<-read.table('ValYHatFromXGB2.csv',header=T)

# log loss function
LL <- function(Pred,YVal){
  ll <- -mean(YVal*log(Pred)+(1-YVal)*log(1-Pred))
  return(ll)
}

# average of all 4 models  
ave_4<-(dt+lg+nn+xgb)/4
LL(ave_4$predTree,YVal)
# log loss = 0.4086

# average of dt lg and nn 
ave_dln<-(dt+lg+nn)/3
LL(ave_dln$predTree,YVal)
# log loss = 0.4145

# average of dt lg and xgb
ave_dlx<-(dt+lg+xgb)/3
LL(ave_dlx$predTree,YVal)
# log loss = 0.4063

# average of dt nn and xgb
ave_dnx<-(dt+nn+xgb)/3
LL(ave_dnx$predTree,YVal)
# log loss = 0.4066

# average of lg nn and xgb
ave_lnx<-(lg+nn+xgb)/3
LL(ave_lnx$s0,YVal)
# log loss = 0.4100

# average of dt and xgb
ave_dx<-(dt+xgb)/2
LL(ave_dx$predTree,YVal)
# log loss = 0.4031


