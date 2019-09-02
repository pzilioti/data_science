library(ggplot2)

library(rpart)
library(rpart.plot)
library(e1071)
library(caret)

heart = read.csv("heart.csv")

#age
#sex
#chest pain type (4 values)
#resting blood pressure
#serum cholestoral in mg/dl
#fasting blood sugar 120 mg/dl
#resting electrocardiographic results (values 0,1,2)
#maximum heart rate achieved
#exercise induced angina
#oldpeak = ST depression induced by exercise relative to rest
#the slope of the peak exercise ST segment
#number of major vessels (0-3) colored by flourosopy
#thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

#limpeza e exploração inicial dos dados
names = c("age", "sex", "chest_pain", "blood_pressure", "cholestoral", "blood_sugar", 
          "electrocardiographic", "heart_rate", "angina", "oldpeak", "slope_st", "major_vessels", "thal", "heart_disease")

colnames(heart) <- names

summary(heart)

heart$sex <- as.factor(heart$sex)
heart$chest_pain <- as.factor(heart$chest_pain)
heart$blood_sugar <- as.factor(heart$blood_sugar)
heart$electrocardiographic <- as.factor(heart$electrocardiographic)
heart$angina <- as.factor(heart$angina)
heart$slope_st <- as.factor(heart$slope_st)
heart$major_vessels <- as.factor(heart$major_vessels)
heart$thal <- as.factor(heart$thal)
heart$heart_disease <- as.factor(heart$heart_disease)

table(is.na(heart))

str(heart)

View(heart)

boxplot(age~heart_disease,data=heart,xlab="Presença de doença no coração", ylab="Idade")
boxplot(blood_pressure~heart_disease,data=heart,xlab="Presença de doença no coração", ylab="Pressão no Sangue")
boxplot(cholestoral~heart_disease,data=heart,xlab="Presença de doença no coração", ylab="Colesterol")

#parece ser uma variavel interessante para a classificação
boxplot(heart_rate~heart_disease,data=heart,xlab="Presença de doença no coração", ylab="Taxa do coração")

ggplot(heart, aes(heart_rate, blood_pressure, colour = heart_disease)) + geom_point()
ggplot(heart, aes(heart_rate, cholestoral, colour = heart_disease)) + geom_point()
ggplot(heart, aes(heart_rate, oldpeak, colour = heart_disease)) + geom_point()

hist(heart$age)
hist(heart$blood_pressure)
hist(heart$cholestoral)
hist(heart$heart_rate)
hist(heart$oldpeak)




#inicio da analise

## 75% of the sample size
smp_size <- floor(0.75 * nrow(heart))

## set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(heart)), size = smp_size)

train <- heart[train_ind, ]
test <- heart[-train_ind, ]

#Analise com arvore de decisao
rtree <- rpart(heart_disease ~ ., train)
rpart.plot(rtree)

pred_tree <- predict(rtree, newdata = test, type = "class")
confusionMatrix(pred_tree,test$heart_disease)

#Analise com SVM
model_svm <- svm(heart_disease ~ ., data=train)
summary(model_svm)

pred_svm <- predict(model_svm,test)
confusionMatrix(pred_svm,test$heart_disease)
