########################AULA 02#####################################
library(ggplot2)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)



heart2 = readRDS("data_heart_train_model.rds")

str(heart2)

heart2$sex <- as.factor(heart2$sex)
heart2$cp <- as.factor(heart2$cp)
heart2$fbs <- as.factor(heart2$fbs)
heart2$restecg <- as.factor(heart2$restecg)
heart2$exang <- as.factor(heart2$exang)
heart2$slope <- as.factor(heart2$slope)
heart2$ca <- as.factor(heart2$ca)
heart2$thal <- as.factor(heart2$thal)
heart2$target <- as.factor(heart2$target)

#graficos
ggplot(heart2, aes(trestbps, chol, colour = target)) + geom_point()



#Verificando se os dados estão balanceados
summary(heart2$target)
table(heart2$target)
# 0   1 
#119 134
#aparentemente está balanceado

#amostras vs variaveis: 253 amostras x 14 variaveis

#verificando null
table(is.na(heart2))


#Split entre treino e teste
set.seed(2019)
indice <- 1:253
totrain <- sample(indice, round(0.7*nrow(heart2),0))
totest <- indice[!(indice %in% totrain)]

train <- heart2[totrain,]
test <- heart2[totest,]


set.seed(3456)

split <- createDataPartition(heart2$target, p = .75, 
                             list = FALSE, 
                             times = 1)
heartTrain <- heart2[ split,]
heartTest  <- heart2[-split,]


#regressão logistica

fit_glm_log <- glm(target ~ ., 
                   data = train, 
                   family = binomial(link = logit))

adj_glm_log <- predict(fit_glm_log, newdata = test, type = "response")



#LDA
fit_lda <- MASS::lda(target ~ ., 
                   data = train)

adj_lda <- predict(fit_lda, newdata = test, type = "response")

test$class_glm_log_50 <- factor(ifelse(adj_glm_log < 0.5,0,1))
test$class_lda_50 <- factor(ifelse(adj_lda$posterior[,2] < 0.5,0,1))

confusionMatrix(test$class_glm_log_50, test$target, positive = "1")
confusionMatrix(test$class_lda_50, test$target, positive = "1")


#Novos dados

dado <-  readRDS("data_heart_without_response.rds")

dado$sex <- as.factor(dado$sex)
dado$cp <- as.factor(dado$cp)
dado$fbs <- as.factor(dado$fbs)
dado$restecg <- as.factor(dado$restecg)
dado$exang <- as.factor(dado$exang)
dado$slope <- as.factor(dado$slope)
dado$ca <- as.factor(dado$ca)
dado$thal <- as.factor(dado$thal)

str(dado)


dado_adj_glm_log <- predict(fit_glm_log, newdata = dado, type = "response")
dado$class_glm_log_50 <- factor(ifelse(dado_adj_glm_log < 0.5,0,1))

summary(dado)

resposta <- readRDS("data_heart_with_response.rds")

resposta$target <- as.factor(resposta$target)

confusionMatrix(dado$class_glm_log_50, resposta$target, positive = "1")



############Naive Bayes####################
library(e1071)


?naiveBayes

fit_naive_bayes <- naiveBayes(target ~ . , train)

summary(fit_naive_bayes)

adj_naive_bayes <- predict(fit_naive_bayes, newdata = test, type = "class")
adj_naive_bayes2 <- predict(fit_naive_bayes, newdata = test, type = "raw")

test$naive_bayes <- adj_naive_bayes

confusionMatrix(test$naive_bayes, test$target, positive = "1")
