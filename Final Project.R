rm(list = ls())
setwd("~/Dropbox/UofT Admin and TA/STA 302/Lectures/Final Project")
dev.off()

install.packages('tidyverse')
install.packages('NHANES')
install.packages('glmnet')
install.packages('rms')

library(NHANES)  ## 34,35,59
library(tidyverse)
library(glmnet)
library(car)
library(rms)
library(grDevices)

par(mfrow = c(1,1), family = 'serif')
small.nhanes <- na.omit(NHANES[NHANES$SurveyYr=="2011_12"
                               & NHANES$Age > 17,c(1,3,4,8:11,13,17,20,21,25,46,50,51,52,61)])
small.nhanes <- as.data.frame(small.nhanes %>%
  group_by(ID) %>% filter(row_number()==1) )

nrow(small.nhanes)

## Checking whether there are any ID that was repeated. If not ##
## then length(unique(small.nhanes$ID)) and nrow(small.nhanes) are same ##
length(unique(small.nhanes$ID))

## Create training and test set ##
set.seed(1005570035)

train <- small.nhanes[sample(seq_len(nrow(small.nhanes)), size = 400),]
nrow(train)

head(train)
length(which(small.nhanes$ID %in% train$ID))
test <- small.nhanes[!small.nhanes$ID %in% train$ID,]
nrow(test)

## check the outliers for training dataset.

Outliers_train = boxplot(train$BPSysAve)$out
boxplot(train$BPSysAve,ylab = "BPSysAve",main = "Boxplot with outliers")
train = train[train$BPSysAve < 164 & train$BPSysAve > 85, ]
boxplot(train$BPSysAve,ylab = "BPSysAve",main = "Boxplot with outliers removed")
Outliers_train = boxplot(train$BPSysAve)$out

str(train)
nrow(train)


## check the outliers for testing dataset. ##
Outliers_test = boxplot(test$BPSysAve)$out
test = test[test$BPSysAve < min(Outliers_test),]
nrow(test)



## check the boxplot for SmokeNow and BPSysAve ##
summary(train$BPSysAve)
box_plot = boxplot(train$BPSysAve ~ train$SmokeNow, xlab = "Whether Smoke or Not", ylab = "BPSysAve",main = "BPSysAve VS Smoking status")

## smokers categorized by gender ##
male_smokers = train[train$Gender == "male" & train$SmokeNow == "Yes",]
female_smokers = train[train$Gender == "female" & train$SmokeNow == "Yes",]

## Non smokers categorized by gender ##
male_non_smokers = train[train$Gender == "male" & train$SmokeNow == "No",]
female_non_smokers = train[train$Gender == "female" & train$SmokeNow == "No",]






## create a two way table ##
smoke <- matrix(c(95,73,168,118,96,214,213,169,382),ncol=3,byrow=TRUE)
colnames(smoke) <- c("Male","Female","Total")
rownames(smoke) <- c("Smoke","Do not Smoke","Total")
smoke <- as.table(smoke)
smoke
smoke/382
barplot(smoke,legend=T,beside=T,main='Smoking Status by Gender')



## Splitting dataset by Gender ##
man_data = train[train$Gender == "male",]
fem_data = train[train$Gender == "female",]

summary(man_data$Age)
summary(fem_data$Age)

mean(male_smokers$Age)
mean(male_non_smokers$Age)


mean(female_smokers$Age)
mean(female_non_smokers$Age)




### Now fit a multiple linear regression ##
model.lm <- lm( BPSysAve ~ ., data = train[, -c(1)])
summary(model.lm)
anova(model.lm)

## Perform Prediction ##
pred.y <- predict(model.lm, newdata = test, type = "response")

## Prediction error ##
mean((test$BPSysAve - pred.y)^2) 

#outlierTest(model.lm)

resid <- rstudent(model.lm)
resid
fitted <- predict(model.lm)

qqnorm(resid)
qqline(resid)

plot(resid ~ fitted, type = "p", xlab = "Fitted Values", 
     ylab = "Standardized Residual", cex.lab = 1.2,
     col = "red")
lines(lowess(fitted, resid), col = "blue") ## There is no evident pattern and hence we do not
## need any transformation.

# Response vs Fitted values ##
plot(train$BPSysAve ~ fitted, type = "p", xlab = "Fitted Values", 
     ylab = "BPSysAve", cex.lab = 1.2,
     col = "red")

abline(lm(train$BPSysAve ~ fitted), lwd = 2, col = "blue")
lines(lowess(fitted, train$BPSysAve), col = "red") # No transformation is needed.

vif(model.lm) ## "Weight" and "BMI" has high VIF, might need to be removed.

## hat values ##
h <- hatvalues(model.lm)
thresh <- 2 * (dim(model.matrix(model.lm))[2])/nrow(train)
w <- which(h > thresh)
w
train[w,]
length(w)


### The Influential Observations ####
D <- cooks.distance(model.lm)

which(D > qf(0.5, length(model.lm$coefficients)-2,  nrow(train) - length(model.lm$coefficients)-2))

## DFFITS ##
dfits <- dffits(model.lm)
d = which(abs(dfits) > (2*(sqrt(length(model.lm$coefficients)-2/nrow(train)))))
d

## DFBETAS ##
dfb <- dfbetas(model.lm)
df = which(abs(dfb[,1]) > (2/sqrt(nrow(train))))
length(df)

## Stepwise Selection ##
## Based on AIC ##

n <- nrow(train)
sel.var.aic <- step(model.lm, trace = 0, k = 2, direction = "both") 
sel.var.aic<-attr(terms(sel.var.aic), "term.labels")   
sel.var.aic

## Based on BIC ##
sel.var.bic = step(model.lm, trace = 0, k = log(n), direction = "both")
sel.var.bic = attr(terms(sel.var.bic), "term.labels")
sel.var.bic



## Fit a ridge penalty ##
model.ridge <- glmnet(x = model.matrix( ~ ., data = train[,-c(1,12)]), y = train$BPSysAve, 
                      standardize = T, alpha = 0)

## Perform Prediction ##
pred.y.ridge <- predict(model.ridge, newx = model.matrix( ~ ., data = test[,-c(1,12)]), type = "response")

## Prediction error ##
mean((test$BPSysAve - pred.y.ridge)^2) 


## Fit a LASSO penalty ##
model.lasso <- glmnet(x = model.matrix( ~ ., data = train[,-c(1,12)]), y = train$BPSysAve
                      , standardize = T, alpha = 1)

## Perform Prediction ##
pred.y.lasso <- predict(model.lasso, newx = model.matrix( ~ ., data = test[,-c(1,12)]), type = "response")
## Prediction error ##
mean((test$BPSysAve - pred.y.lasso)^2) #256

## Elastic net ##

model.EN <- glmnet(x = model.matrix( ~ ., data = train[,-c(1,12)]), y = train$BPSysAve, standardize = T, alpha = 0.5)

## Perform Prediction ##
pred.y.EN <- predict(model.EN, newx = model.matrix( ~ ., data = test[,-c(1,12)]), type = "response")

## Prediction error ##
mean((test$BPSysAve - pred.y.EN)^2) # 256

### LASSO selection ###

## Perform cross validation to choose lambda ##
set.seed(1005570035)
cv.out <- cv.glmnet(x = model.matrix( ~ ., data = train[,-c(1,12)]), y = train$BPSysAve, standardize = T, alpha = 1)
plot(cv.out)
best.lambda <- cv.out$lambda.1se
best.lambda
co<-coef(cv.out, s = "lambda.1se")

#Selection of the significant features(predictors)

## threshold for variable selection ##

thresh <- 0.00
# select variables #
inds<-which(abs(co) > thresh )
variables<-row.names(co)[inds]
sel.var.lasso<-variables[!(variables %in% '(Intercept)')]
sel.var.lasso

### Cross Validation and prediction performance of AIC based selection ###
ols.aic <- ols(BPSysAve ~ ., data = train[,which(colnames(train) %in% c(sel.var.aic, "BPSysAve"))], 
               x=T, y=T, model = T)

## 10 fold cross validation ##    
aic.cross <- calibrate(ols.aic, method = "crossvalidation", B = 10)

## Calibration plot ##

plot(aic.cross, las = 1, xlab = "Predicted Probability", main = "Cross-Validation calibration with AIC")


## Test Error ##
pred.aic <- predict(ols.aic, newdata = test[,which(colnames(train) %in% c(sel.var.aic, "BPSysAve"))])

## Prediction error ##
pred.error.AIC <- mean((test$BPSysAve - pred.aic)^2)
pred.error.AIC

### Cross Validation and prediction performance of BIC based selection ###
ols.bic <- ols(BPSysAve ~ ., data = train[,which(colnames(train) %in% c(sel.var.bic, "BPSysAve"))], 
               x=T, y=T, model = T)

## 10 fold cross validation ##    
bic.cross <- calibrate(ols.bic, method = "crossvalidation", B = 10)


## Calibration plot ##

plot(bic.cross, las = 1, xlab = "Predicted Probability", main = "Cross-Validation calibration with BIC")


## Test Error ##
pred.bic <- predict(ols.bic, newdata = test[,which(colnames(train) %in% c(sel.var.bic, "BPSysAve"))])

## Prediction error ##
pred.error.BIC <- mean((test$BPSysAve - pred.bic)^2) ## 235

### Cross Validation and prediction performance of lasso based selection ###
ols.lasso <- ols(BPSysAve ~ ., data = train[,which(colnames(train) %in% c(sel.var.lasso, "BPSysAve"))], 
                 x=T, y=T, model = T)

## 10 fold cross validation ##    
lasso.cross <- calibrate(ols.lasso, method = "crossvalidation", B = 10)
## Calibration plot ##

plot(lasso.cross, las = 1, xlab = "Predicted Probability", main = "Cross-Validation calibration with LASSO")


## Test Error ##
pred.lasso <- predict(ols.lasso, newdata = test[,which(colnames(train) %in% c(sel.var.lasso, "BPSysAve"))])
head(pred.lasso)

## Prediction error ##
pred.error.lasso <- mean((test$BPSysAve - pred.lasso)^2)

print(c(pred.error.AIC, pred.error.BIC, pred.error.lasso)) ## 235 ## 154 when test outliers removed ## 162 when train outliers removed


## scatterplot of age vs bpsAve categorized by smoking status ##

plot(train$BPSysAve ~ train$Age, type = "p", xlab = "Age",ylab = "BPSysAve",
     col = ifelse(train$Gender == 'male', "red", "blue"))
abline(lm(BPSysAve ~ Age,data = train[train$SmokeNow == 'Yes',]),col = "red")
abline(lm(BPSysAve ~ Age,data = train[train$SmokeNow == 'No',]),col = "blue")

legend("topleft", legend = c("Smokers", "Non-Smokers"),
       col = c("red", "blue"), lty = 1:2, cex = 0.6)

## scatterplot of age vs bpsAve according to gender ##
par(mfrow = c(1,1), family = 'serif')
plot(train$BPSysAve ~ train$Age, type = "p", xlab = "Age",ylab = "BPSysAve",
     col = ifelse(train$Gender == 'male', "red", "blue"))
abline(lm(BPSysAve ~ Age,data = train[train$Gender == 'male',]),col = "red")
abline(lm(BPSysAve ~ Age,data = train[train$Gender == 'female',]),col = "blue")

legend("topleft", legend = c("male", "female"),
       col = c("red", "blue"), lty = 1:2, cex = 0.6)




## Final model
fin.model = lm(BPSysAve ~ Age + Gender, data = train ) #gender+age 18,148.4
summary(fin.model)
confint(fin.model)
anova(fin.model)
pred.fin <- predict(fin.model, newdata = test, type = "response")



## Prediction error ##
mean((test$BPSysAve - pred.fin)^2)


## interaction model
#model <- lm(BPSysAve ~ Age + as.factor(SmokeNow) + BMI*as.factor(SmokeNow), data = train)
#summary(model)
#anova(model)

