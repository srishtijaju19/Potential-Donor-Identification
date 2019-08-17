library(plyr)
library(dplyr)
library(psych)
library(rpart)
library(rpart.plot)
library(ROCR)
library(heuristica)
library(lift)
library(gains)
library(C50)
library(utils)
library(csv)
library(csvread)

#Read Data
dTble <- read.csv(file = "Dataset_2.csv", header = TRUE, sep = ",")
 
#converting to factor
dTble[,c(4)] <- as.factor(dTble[,c(4)])
dTble[,c(6)] <- as.factor(dTble[,c(6)])
dTble[,c(8)] <- as.factor(dTble[,c(8)])
dTble[,c(164)] <- as.factor(dTble[,c(164)])

#Removing Cat variables
dTble_c <- as.data.frame(dTble[,c(1,3,4,5,6,8,164)])
dtble_n <- as.data.frame(dTble[,c(-1,-3,-4,-5,-6,-8,-164)])

#Building Correlation Matrix
cor_matrix <- abs(cor(dtble_n))
diag(cor_matrix) <- 0
library(corrplot)
corrplot(cor_matrix, method = "square",tl.cex = 0.4)

#Principal Component Analysis for data reduction
prCmpA <- prcomp(dtble_n,scale. = TRUE)
stdev_pr <- prCmpA$sdev
var_pr <- stdev_pr^2
cumvar_pr <- var_pr/sum(var_pr)
sum(cumvar_pr[1:20])

plot(cumsum(cumvar_pr), xlab = "Prinicipal component", ylab = "Cumulative Variance", type = "b")
abline(h=0.75,col = "red",v = 20)

summary(prCmpA)

#--------------------- Combining PCA with Variables ---------------------------------

temp1 <- prCmpA$x
dTble_new <- cbind(temp1[,c(1:20)],dTble_c)
#dTble_new[,c(21:26)] <- NULL/

#--------------------- Building DT to measure Variable Importance -------------------

library(caret)

Var_dT <- rpart(TARGET_B ~ .,data = dTble[,c(-1,-164)], method = "class", parms = list(prior = c(0.6,0.4), loss = matrix(c(0,2,4,0), byrow = TRUE, nrow = 2),split = "gini"),
                  control = rpart.control(minsplit = 200, minbucket = round(200/3), cp = 0.001, maxcompete = 4, 
                                          maxsurrogate = 5, usesurrogate = 2, xval = 10, surrogatestyle = 0, maxdepth = 15))


t <- as.data.frame(varImp(Var_dT))

#------------------------- Modelling ------------------------------------------

#Building models with 60% training data

set.seed(12345)

dTrn_dTble <- sample_frac(dTble_new, 0.6)
dTst_dTble <- setdiff(dTble_new,dTrn_dTble)

#------------------------- Decision Tree --------------------------------------

library(caret)

model_dT <- rpart(TARGET_B ~ .,data = dTrn_dTble[,c(-21)], method = "class", parms = list(prior = c(0.55,0.45), loss = matrix(c(0,1,2,0), byrow = TRUE, nrow = 2),split = "gini"),
                  control = rpart.control(minsplit = 170, minbucket = round(170/3), cp = 0.0005, maxcompete = 4, 
                                    maxsurrogate = 5, usesurrogate = 2, xval = 8, surrogatestyle = 0, maxdepth = 15))

#rpart.plot(model_dT,box.palette = "GnRd", shadow.col = "gray",nn = TRUE, cex = 0.7,roundint = FALSE)
plotcp(model_dT)

prediction_dT <- predict(model_dT,newdata = dTst_dTble[,c(-21,-27)])

colnames(prediction_dT) <- c("c_0","c_1")

prediction_dT1 <- prediction_dT %>% 
  as.data.frame() %>% 
  mutate(value = ifelse(c_1 > 0.5,1,0))

temp3 <- prediction(prediction_dT1[,c(3)],dTst_dTble[,c(27)])

prediction_dT1_PR <- performance(temp3,"prec","rec")
plot(prediction_dT1_PR)

prediction_dT1_F1 <- performance(temp3,"f")
print(prediction_dT1_F1@y.values)

plotLift(as.factor(prediction_dT1[,c(3)]),dTst_dTble[,c(27)])

confusionMatrix(as.factor(prediction_dT1[,c(3)]),dTst_dTble[,c(27)])

library(AUC)

auc_1 <- auc(roc(as.factor(prediction_dT1[,c(3)]),dTst_dTble[,c(27)])) 
print(auc_1)

#----------------------- Profit Curve for Decision Trees ----------------------

profit_table_dt <- as.data.frame(prediction_dT1[,c(2:3)])
profit_table_dt[,c(2)] <- as.factor(profit_table_dt[,c(2)])

colnames(profit_table_dt) <- c("Probability","Prediction")

profit_table_dt[,"label"] <- dTst_dTble[,c(27)]

profit_table_dt <- profit_table_dt %>%
  arrange(desc(Probability)) %>% 
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.73,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_dt[,"cum_profit"] <- cumsum(profit_table_dt$profit)

plot(profit_table_dt$cum_profit)
Opt_thres_DT <- profit_table_dt[which.max(profit_table_dt$cum_profit),1]

#------------------------- Logistic Regression --------------------------------

dTrn_dTble <- dTrn_dTble[,c(-21)]
dTst_dTble <- dTst_dTble[,c(-21)]


dTst_dTble <- dTst_dTble %>% 
  mutate(GENDER1 = ifelse(GENDER == "A", "M",as.character(GENDER))) %>% 
  select("PC1","PC2","PC3","PC4","PC5" , "PC6" ,"PC7","PC8" ,"PC9","PC10" ,"PC11","PC12", "PC13" ,"PC14","PC15","PC16" ,"PC17" , "PC18",
         "PC19","PC20" , "HOMEOWNR", "INCOME","GENDER1","WEALTH1","WEALTH2","TARGET_B")

dTst_dTble[,c('GENDER1')] <- as.factor(dTst_dTble[,c('GENDER1')])
colnames(dTst_dTble) <- c("PC1","PC2","PC3","PC4","PC5" , "PC6" ,"PC7","PC8" ,"PC9","PC10" ,"PC11","PC12", "PC13" ,"PC14","PC15","PC16" ,"PC17" , "PC18",
                          "PC19","PC20" , "HOMEOWNR", "INCOME","GENDER","WEALTH1","WEALTH2","TARGET_B")

dTrn_dTble <- dTrn_dTble %>% 
  mutate(GENDER1 = ifelse(GENDER == "A", "M",as.character(GENDER))) %>% 
  select("PC1","PC2","PC3","PC4","PC5" , "PC6" ,"PC7","PC8" ,"PC9","PC10" ,"PC11","PC12", "PC13" ,"PC14","PC15","PC16" ,"PC17" , "PC18",
         "PC19","PC20" , "HOMEOWNR", "INCOME","GENDER1","WEALTH1","WEALTH2","TARGET_B")

dTrn_dTble[,c('GENDER1')] <- as.factor(dTrn_dTble[,c('GENDER1')])
colnames(dTrn_dTble) <- c("PC1","PC2","PC3","PC4","PC5" , "PC6" ,"PC7","PC8" ,"PC9","PC10" ,"PC11","PC12", "PC13" ,"PC14","PC15","PC16" ,"PC17" , "PC18",
                          "PC19","PC20" , "HOMEOWNR", "INCOME","GENDER","WEALTH1","WEALTH2","TARGET_B")

library(glmnet)

## Logistice Regression without Lasso/Ridge
glm_model <- glm(TARGET_B ~ .,data = dTrn_dTble, family = binomial)
summary(glm_model)

glm_prob <- predict.glm(glm_model,dTst_dTble[,c(-26)],type="response")
glm_prob1 <- as.data.frame(glm_prob)

colnames(glm_prob1) <- c("pos")

glm_class <- glm_prob1 %>% 
  as.data.frame() %>% 
  mutate(value = ifelse(pos > 0.243,1,0))

confusionMatrix(as.factor(glm_class[,c(2)]),dTst_dTble[,c(26)])


## Logistice Regression with Lasso
x <- model.matrix(TARGET_B ~ ., data = dTrn_dTble)

cv_lm <- cv.glmnet(x,dTrn_dTble[,c(26)], family = "binomial", type.measure = "mse",alpha=1)
plot(cv_lm)

#min value of lambda
lambda_min <- cv_lm$lambda.min
#best value of lambda
lambda_1se <- cv_lm$lambda.1se
#regression coefficients
coef(cv_lm,s=lambda_1se)


#get test data
x_test <- model.matrix(TARGET_B~.,dTst_dTble) 

#predict class, type="class"
lasso_prob <- predict(cv_lm,newx = x_test,s=lambda_min,type="response")

prediction_laslg <- as.data.frame(lasso_prob)

colnames(prediction_laslg) <- c("pos")

min_p <- which.min(prediction_laslg[,c(1)])
max_p <- which.max(prediction_laslg[,c(1)])

Pr_min <- prediction_laslg[c(min_p),]
Pr_max <- prediction_laslg[c(max_p),]

prediction_laslg2 <- prediction_laslg %>% 
  mutate(p_scale = (pos - Pr_min) / (Pr_max - Pr_min))

prediction_laslg2 <- prediction_laslg2 %>% 
  mutate(value = ifelse(p_scale > 0.438,1,0))

confusionMatrix(as.factor(prediction_laslg2[,c(3)]),dTst_dTble[,c(26)])

plotLift(as.factor(prediction_laslg2[,c(3)]),dTst_dTble[,c(26)])

#----------------- Profit Curve for Lasso --------------------------------------

profit_table_ls <- as.data.frame(prediction_laslg2[,c(2:3)])
profit_table_ls[,c(2)] <- as.factor(profit_table_ls[,c(2)])

colnames(profit_table_ls) <- c("Probability","Prediction")

profit_table_ls[,"label"] <- dTst_dTble[,c(26)]

profit_table_ls <- profit_table_ls %>%
  arrange(desc(Probability)) %>% 
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.03,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_ls[,"cum_profit"] <- cumsum(profit_table_ls$profit)

plot(profit_table_ls$cum_profit)
Opt_thres_LS <- profit_table_ls[which.max(profit_table_ls$cum_profit),1]

#-------------------------------------------------------------------------------

## Logistice Regression with Ridge
x <- model.matrix(TARGET_B ~ ., data = dTrn_dTble)

cv_lm <- cv.glmnet(x,dTrn_dTble[,c(26)], family = "binomial", type.measure = "mse",alpha=0)
plot(cv_lm)

#min value of lambda
lambda_min <- cv_lm$lambda.min
#best value of lambda
lambda_1se <- cv_lm$lambda.1se
#regression coefficients
coef(cv_lm,s=lambda_1se)


#get test data
x_test <- model.matrix(TARGET_B~.,dTst_dTble) 

#predict class, type="class"
rid_prob <- predict(cv_lm,newx = x_test,s=lambda_1se,type="response")

prediction_ridlg <- as.data.frame(rid_prob)

colnames(prediction_ridlg) <- c("pos")

min_p <- which.min(prediction_ridlg[,c(1)])
max_p <- which.max(prediction_ridlg[,c(1)])

Pr_min <- prediction_ridlg[c(min_p),]
Pr_max <- prediction_ridlg[c(max_p),]

prediction_ridlg2 <- prediction_ridlg %>% 
  mutate(p_scale = (pos - Pr_min) / (Pr_max - Pr_min))

prediction_ridlg2 <- prediction_ridlg2 %>% 
  mutate(value = ifelse(p_scale > 0.5,1,0))

confusionMatrix(as.factor(prediction_ridlg2[,c(3)]),dTst_dTble[,c(26)])

plotLift(as.factor(prediction_ridlg2[,c(3)]),dTst_dTble[,c(26)])

#----------------------- Profit Curve for Ridge --------------------------------------

profit_table_rd <- as.data.frame(prediction_ridlg2[,c(2:3)])
profit_table_rd[,c(2)] <- as.factor(profit_table_rd[,c(2)])

colnames(profit_table_rd) <- c("Probability","Prediction")

profit_table_rd[,"label"] <- dTst_dTble[,c(26)]

profit_table_rd <- profit_table_rd %>%
  arrange(desc(Probability)) %>% 
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.73,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_rd[,"cum_profit"] <- cumsum(profit_table_rd$profit)

plot(profit_table_rd$cum_profit)
Opt_thres_RD <- profit_table_rd[which.max(profit_table_rd$cum_profit),1]

#------------------------------ Boosted Trees ----------------------------------------

library(gbm)

model_bt <- gbm(TARGET_B ~ .,data = dTrn_dTble,distribution = "gaussian", n.trees = 2000, bag.fraction = 0.8, cv.folds = 8, interaction.depth = 4, shrinkage = 0.01)

gbm.perf(model_bt, method = "cv")

prediction_bt <- predict(model_bt,newdata = dTst_dTble[,c(-26)], type = "response")
prediction_bt1 <- as.data.frame(prediction_bt)

min_p <- which.min(prediction_bt1[,c(1)])
max_p <- which.max(prediction_bt1[,c(1)])

Pr_min <- prediction_bt1[c(min_p),]
Pr_max <- prediction_bt1[c(max_p),]

colnames(prediction_bt1) <- c("pos")

prediction_bt2 <- prediction_bt1 %>% 
    mutate(p_scale = (pos - Pr_min) / (Pr_max - Pr_min))

prediction_bt2 <- prediction_bt2 %>% 
  mutate(value = ifelse(p_scale > 0,1,0))

temp5 <- prediction(prediction_bt2[,c(3)],dTst_dTble[,c(26)])

prediction_bt_PR <- performance(temp5,"prec","rec")
#plot(prediction_bt_PR)

prediction_bt_F1 <- performance(temp5,"f")
print(prediction_bt_F1@y.values)

plotLift(as.factor(prediction_bt2[,c(3)]),dTst_dTble[,c(26)])

prediction_bt2[,c(3)] <- as.factor(prediction_bt2[,c(3)])

confusionMatrix(as.factor(prediction_bt2[,c(3)]),dTst_dTble[,c(26)])

#----------------------- Profit Curve for Boosting --------------------------------------

profit_table_bt <- as.data.frame(prediction_bt2[,c(2:3)])
profit_table_bt[,c(2)] <- as.factor(profit_table_bt[,c(2)])

colnames(profit_table_bt) <- c("Probability","Prediction")

profit_table_bt[,"label"] <- dTst_dTble[,c(26)]

profit_table_bt <- profit_table_bt %>%
  arrange(desc(Probability)) %>%
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.73,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_bt[,"cum_profit"] <- cumsum(profit_table_bt$profit)

plot(profit_table_bt$cum_profit)
Opt_thres_BT <- profit_table_bt[which.max(profit_table_bt$cum_profit),1]

#------------------------- Random Forrest -----------------------------------------------

library(randomForest)

model_rf <- randomForest(TARGET_B ~ .,data = dTrn_dTble, ntree = 22, maxnodes = 10, importance = TRUE, proximity=TRUE, oob.prox=TRUE)
summary(model_rf)

prediction_rf <- predict(model_rf,newdata = dTst_dTble[,c(-26)],type = "prob")

prediction_rf1 <- as.data.frame(prediction_rf[,c(2)])

min_p <- which.min(prediction_rf1[,c(1)])
max_p <- which.max(prediction_rf1[,c(1)])

Pr_min <- prediction_rf1[c(min_p),]
Pr_max <- prediction_rf1[c(max_p),]

colnames(prediction_rf1) <- c("pos")

prediction_rf2 <- prediction_rf1 %>% 
  mutate(p_scale = (pos - Pr_min) / (Pr_max - Pr_min))

prediction_rf2 <- prediction_rf2 %>% 
  mutate(value = ifelse(p_scale > 0,1,0))

temp4 <- prediction(prediction_rf2[,c(3)],dTst_dTble[,c(26)])

prediction_rf1_PR <- performance(temp4,"prec","rec")
plot(prediction_rf1_PR)

prediction_rf1_F1 <- performance(temp4,"f")
print(prediction_rf1_F1@y.values)

plotLift(as.factor(prediction_rf2[,c(3)]),dTst_dTble[,c(26)])

confusionMatrix(as.factor(prediction_rf2[,c(3)]),dTst_dTble[,c(26)])

getTree(model_rf,k=9, labelVar = TRUE)

varImpPlot(model_rf)
plot(margin(model_rf))

library(party)

x <- ctree(TARGET_B ~ .,data = dTrn_dTble)
plot(x,type = "simple")

#----------------------- Profit Curve for Random Forest --------------------------------------

profit_table_rf <- as.data.frame(prediction_rf2[,c(2:3)])
profit_table_rf[,c(2)] <- as.factor(profit_table_rf[,c(2)])

colnames(profit_table_rf) <- c("Probability","Prediction")

profit_table_rf[,"label"] <- dTst_dTble[,c(26)]

profit_table_rf <- profit_table_rf %>%
  arrange(desc(Probability)) %>%
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.73,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_rf[,"cum_profit"] <- cumsum(profit_table_rf$profit)

plot(profit_table_rf$cum_profit)
Opt_thres_RF <- profit_table_rf[which.max(profit_table_rf$cum_profit),1]

#----------------------------- Support Vector Machines --------------------------------------

library(e1071)

model_svm <- svm(TARGET_B ~ ., data = dTrn_dTble, kernel = "sigmoid", gamma = 2, coef0 = 1,
                 cost = 0.001, tolerance =  0.1, epsilon = 0.001, shrinking = TRUE, probability = TRUE)

prediction_svm <- predict(model_svm,newdata = dTst_dTble[,c(-26)], decision.values = TRUE, probability = TRUE)
prediction_svm1 <- as.data.frame(attr(prediction_svm,"probabilities"))

min_p <- which.min(prediction_svm1[,c(1)])
max_p <- which.max(prediction_svm1[,c(1)])

Pr_min <- prediction_svm1[c(min_p),]
Pr_max <- prediction_svm1[c(max_p),]

colnames(prediction_svm1) <- c("c_1","c_0")

prediction_svm2 <- prediction_svm1 %>% 
  as.data.frame() %>%
  mutate(pos = (c_1 - Pr_min[1,1]) / (Pr_max[1,1] - Pr_min[1,1]))

prediction_svm2[,c(2)] <- NULL

prediction_svm2 <- prediction_svm2 %>% 
  mutate(value = ifelse(pos > 0.5,1,0))

temp8 <- prediction(as.numeric(prediction_svm2[,c(3)]),dTst_dTble[,c(26)])

prediction_svm1_PR <- performance(temp8,"prec","rec")
plot(prediction_svm1_PR)

prediction_svm1_F1 <- performance(temp8,"f")
print(prediction_svm1_F1@y.values)

plotLift(as.factor(prediction_svm2[,c(3)]),dTst_dTble[,c(26)])

confusionMatrix(as.factor(prediction_svm2[,c(3)]),dTst_dTble[,c(26)])

#----------------------- Profit Curve for Support Vector Machines ---------------------------------

profit_table_sv <- as.data.frame(prediction_svm2[,c(2:3)])
profit_table_sv[,c(2)] <- as.factor(profit_table_sv[,c(2)])

colnames(profit_table_sv) <- c("Probability","Prediction")

profit_table_sv[,"label"] <- dTst_dTble[,c(26)]

profit_table_sv <- profit_table_sv %>%
  arrange(desc(Probability)) %>%
  as.data.frame() %>%
  mutate(profit = ifelse((Prediction == 1) & (label == 1),2.73,ifelse((Prediction == 1) & (label == 0),-0.84,0)))

profit_table_sv[,"cum_profit"] <- cumsum(profit_table_sv$profit)

plot(profit_table_sv$cum_profit)
Opt_thres_SV <- profit_table_sv[which.max(profit_table_sv$cum_profit),1]

#--------------------------Predicting Donation Amount ---------------------------------------------

