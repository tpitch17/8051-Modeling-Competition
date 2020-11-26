library(dplyr)
library(data.table)
library(ggplot2)
library(tweedie)
library(statmod)
library(caret)
library(TDboost)

####Exploratory analysis####
#Read in the data
train <- read.csv("Data/InsNova_train.csv", header=TRUE)
test <- read.csv("Data/InsNova_test.csv", header = TRUE)

#Notice that the factors are character types, need to change
sapply(train, class)
sapply(test, class)

#This will change characters to factors
train <- train %>% mutate_if(sapply(train, is.character), as.factor)
test <- test %>% mutate_if(sapply(test, is.character), as.factor)
#sapply(train, class)

#Number of policies
nrow(train)

#Number of policies with a claim
sum(train$claim_ind)

#This was messy, gonna look pairwise
#ggpairs(train, title="Scatter plot Matrix of InsNova") 

ggplot(train, aes(x=veh_value, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

ggplot(train, aes(x=exposure, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

ggplot(train, aes(x=veh_body, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()
#Number of cars of each body type
train %>% group_by(veh_body) %>%
  summarise(nrows = length(veh_body))

ggplot(train, aes(x=veh_age, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

#Number of cars of each age group
train %>% group_by(veh_age) %>%
  summarise(nrows = length(veh_age))

ggplot(train, aes(x=gender, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

ggplot(train, aes(x=area, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

ggplot(train, aes(x=dr_age, y=claim_cost)) + geom_bin2d(bins = 70) +
  scale_fill_continuous(type = "viridis") +
  theme_bw()

####Basic Models####
#Just a basic linear model
lmod <- lm(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
           + area + dr_age, data=train)

#Model with all interactions, doesn't seem useful
fullmod <- lm(claim_cost ~ veh_value*exposure*veh_body*veh_age*gender
           *area*dr_age, data=train)

#Basic logistic of claim indicator
logmod <- glm(claim_ind ~ veh_value + exposure + veh_body + veh_age + gender
              + area + dr_age, data=train, family=binomial(link="logit"))

#Testing out a tweedie
tweemod <- glm(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
               + area + dr_age, data=train, family=tweedie(var.power=1.5))

####Trying out TDboost####
TDboost1 <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
             + area + dr_age, data=train, cv.folds = 5,
             distribution = list(name="EDM", alpha=2))


best.iter <- TDboost.perf(TDboost1,method="cv")
print(best.iter)

#Some predictions
test.predict <- predict.TDboost(TDboost1,test,best.iter)
train.predict <- predict.TDboost(TDboost1,train,best.iter)

summary(TDboost1,n.trees=1) # based on the first tree
summary(TDboost1,n.trees=best.iter) # based on the estimated best number of trees

#least squares error on training data
print(sum((train$claim_cost-train.predict)^2))

#least squares error for tweedie
twtrain.predict <- predict.glm(tweemod, train)
print(sum((train$claim_cost-twtrain.predict)^2))


normalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    population.delta <- 1 / length(a)
    total.losses <- sum(a)
    null.losses <- rep(population.delta, length(a)) # Hopefully is similar to accumulatedPopulationPercentageSum
    accum.losses <- temp.df$actual / total.losses # Hopefully is similar to accumulatedLossPercentageSum
    gini.sum <- cumsum(accum.losses - null.losses) # Not sure if this is having the same effect or not
    sum(gini.sum) / length(a)
  }
  Gini(aa,pp) / Gini(aa,aa)
}

#Test gini out
normalizedGini(train$claim_cost, train.predict)
normalizedGini(train$claim_cost, twtrain.predict)

#Let's test out a submission
View(train.predict)
submission <- cbind(test, test.predict)
submission <- submission[c(1,9)]
names(submission) <- c("id", "claim_cost")
write.csv(submission, file="11-18Test.csv")

##Testing out more TDboost 
TDboost2 <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age, data=train, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.5),
                    interaction.depth = 3)


best.iter2 <- TDboost.perf(TDboost2,method="cv")
print(best.iter2)
train.predict2 <- predict.TDboost(TDboost2, train, best.iter2)
normalizedGini(train$claim_cost, train.predict2)

#Two step TD, first predict claim indicator, then predict claim cost

ind <- TDboost(claim_ind ~ veh_value + exposure + veh_body + veh_age + gender
               + area + dr_age, data=train, cv.folds = 5,
               distribution = list(name="EDM", alpha=2))

best.ind <- TDboost.perf(ind,method="cv")

test$claim_ind <- predict.TDboost(ind, test, best.ind)

cost <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                + area + dr_age + claim_ind, data=train, cv.folds = 5,
                distribution = list(name="EDM", alpha=1.5))

best.cost<- TDboost.perf(cost,method="cv")

test$claim_cost <- predict.TDboost(cost, test, best.cost)

#Next test, logistic regression for indicator, then choose cutoff the model proportion of policies with claims in the train
#Apply cutoff to the predictions done on the test data, then do a cost TDboost that includes the new predictions. Also try with cost with tweedie.

#Proportion of train data with claims
sum(train$claim_ind)/nrow(train)

steps <- seq(0,1,.001)

countind <- function(x){
  sum(ifelse(predict(logmod, train, type = "response")>x,1,0))
}

success_rate <- sapply(steps, countind)
View(cbind(steps, success_rate)) #.135 seems to be the best cutoff

test$claim_ind <- ifelse(predict(logmod, test, type = "response")>.135,1,0)

best.cost<- TDboost.perf(cost,method="cv")

test$claim_costTD <- predict.TDboost(cost, test, best.cost)

twee2 <- glm(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
               + area + dr_age + claim_ind, data=train, family=tweedie(var.power=1.1))

test$claim_costTwee <- predict(twee2, test, type = "response")

Nov25sub <- data.frame("id" = 1:nrow(test), "claim_cost" =  test$claim_costTwee)
write.csv(Nov25sub, file="11-25Test.csv", row.names = FALSE)

Nov25sub2 <- data.frame("id" = 1:nrow(test), "claim_cost" =  test$claim_costTD)
write.csv(Nov25sub2, file="11-25Test2.csv", row.names = FALSE)

######Going at it again with the TDboost
TDboost3 <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age, data=train, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.5), n.trees = 300,
                    interaction.depth = 20)

best.iter3 <- TDboost.perf(TDboost3,method="cv")
summary(TDboost3,n.trees=1)                         
summary(TDboost3,n.trees=best.iter3)

train.predict <- predict.TDboost(TDboost3, train, best.iter3)
print(sum((train$claim_cost - train.predict)^2))
normalizedGini(train$claim_cost, train.predict)

Nov25sub3 <- data.frame("id" = 1:nrow(test), "claim_cost"= predict.TDboost(TDboost3, test, best.iter3))
write.csv(Nov25sub3, file = "11-25Test3.csv", row.names = FALSE)

#Gonna have another go at it

TDboost4 <- TDboost(claim_cost ~ veh_value + exposure + veh_body + veh_age + gender
                    + area + dr_age, data=train, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 300,
                    shrinkage = 0.001, 
                    interaction.depth = 20)

best.iter4 <- TDboost.perf(TDboost4,method="cv")
summary(TDboost4,n.trees=1)                         
summary(TDboost4,n.trees=best.iter4)

train.predict4 <- predict.TDboost(TDboost4, train, best.iter4)
print(sum((train$claim_cost - train.predict4)^2))
normalizedGini(train$claim_cost, train.predict4)

Nov25sub4 <- data.frame("id" = 1:nrow(test), "claim_cost"= predict.TDboost(TDboost4, test, best.iter4))
write.csv(Nov25sub4, file = "11-25Test4.csv", row.names = FALSE)

#Gonna mess around with offset
#This one not good
TDboost5 <- TDboost(claim_cost ~ offset(exposure) + veh_value + veh_body + veh_age + gender
                    + area + dr_age, data=train, cv.folds = 5,
                    distribution = list(name="EDM", alpha=1.7), n.trees = 300,
                    shrinkage = 0.001, 
                    interaction.depth = 20)

best.iter5 <- TDboost.perf(TDboost5,method="cv")
summary(TDboost5,n.trees=1)                         
summary(TDboost5,n.trees=best.iter5)

train.predict5 <- predict.TDboost(TDboost5, train, best.iter5)
print(sum((train$claim_cost - train.predict5)^2))
normalizedGini(train$claim_cost, train.predict4)

Nov25sub5 <- data.frame("id" = 1:nrow(test), "claim_cost"= predict.TDboost(TDboost5, test, best.iter5)+test$exposure)
write.csv(Nov25sub5, file = "11-25Test5.csv", row.names = FALSE)
