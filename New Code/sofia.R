library(RSofia)
library(ModelMetrics)
library(caret)
data_train <- read.csv("/home/harsh/Desktop/Online Shopper's Intention/New Code/data_train_wnf.csv")
data_val <- read.csv("/home/harsh/Desktop/Online Shopper's Intention/New Code/data_val_wnf.csv")
data_test <- read.csv("/home/harsh/Desktop/Online Shopper's Intention/New Code/data_test_wnf.csv")
View(data_train)

{
# spec = c(train = .7, validate = .3)
# g = sample(cut(
#   seq(nrow(data)), 
#   nrow(data)*cumsum(c(0,spec)),
#   labels = names(spec)
# ))
# res = split(data, g)
# data_train <- res$train 
# data_val <- res$validate 
# pp = preProcess(data_train, method = "range")
# data_train <- predict(pp,data_train)
# pp = preProcess(data_val, method="range")
# data_val <- predict(pp,data_val)
# ltype <- c("pegasos","sgd-svm","passive-aggressive","margin-perceptron","romma","logreg-pegasos")
# lptype <- c("roc","combined-roc")
# predtype <- c("linear","logistic")
}

# y_train <- data_train$Revenue
# y_val <- data_val$Revenue
# # y_test <- data_test$Revenue
# data_train <- log(data_train + 1.0)
# data_val <- log(data_val + 1.0)
# data_test <- log(data_test + 1.0)
# data_train$Revenue <- y_train
# data_val$Revenue <- y_val

model <- sofia(Revenue~a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+p1+p2+p3+p4+p5+p6,data=data_train,iterations=1e9,
               learner_type="passive-aggressive",
               loop_type = "combined-roc", verbose = TRUE , random_seed=690, buffer_mb = 2048 )
model$training_time
yhval <- predict(model,newdata = data_val,prediction_type = pt)
#yhval
aucsc <- auc(data_val$Revenue,yhval)
aucsc


# auc(data_val$Revenue,yhval)
