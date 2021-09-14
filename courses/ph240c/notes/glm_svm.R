# GLM/SVM practice

if (!require("e1071")){
  install.packages("e1071")
  libary("e1071")
}
if (!require("glmnet")){
  install.packages("glmnet")
  library("glmnet")
}  
if (!require("dplyr")){
  install.packages("dplyr")
  library("dplyr")
}

# iris Data: Iris, introduced by Ronald Fisher in his 1936 paper
# "The use of multiple measurements in taxonomic problems", 
# contains three plant species (setosa, virginica, versicolor) 
# and four features measured for each sample. 
# These quantify the morphologic variation of the iris flower in its three species,
# all measurements given in centimeters. 

# View(iris)

cdata <- iris %>% 
  dplyr::filter(Species %in% c("versicolor", "virginica")) %>%
  mutate(label = as.integer(Species == "versicolor")) %>%
  select(-Species)
x <- subset(cdata, select = -label)
y <- cdata$label


# SVM-linear kernel
model_svm_l <- svm(as.factor(label) ~ ., data = cdata, kernel = "linear")
summary(model_svm_l)

y_pred_svm_l <- predict(model_svm_l, x)

table(y_pred_svm_l, y)

# SVM-gaussian kernel
model_svm_g <- svm(as.factor(label) ~ ., data = cdata)
summary(model_svm_g)

y_pred_svm_g <- predict(model_svm_g, x)

table(y_pred_svm_g, y)


# SVM-gaussian kernel-soft score
model_svm_c <- svm(as.factor(label) ~ ., data = cdata, cost = 10)
summary(model_svm_c)

y_pred_svm_c <- predict(model_svm_c, x)

table(y_pred_svm_c, y)



# GLM
model_glm <- glm(label ~ ., data = cdata, family = "binomial")
summary(model_glm)

p_pred_glm <- predict(model_glm, x, type = "response")
y_pred_glm <- as.integer(p_pred_glm >= 0.5)
# using "y_pred_glm <- predict(model_glm, x)" will only lead to results of linear predictors

table(y_pred_glm, y)

# GLM with LASSO

model_glmnet <- cv.glmnet(as.matrix(x), y, family = "binomial")
plot(model_glmnet)
coef(model_glmnet)
p_pred_glmnet <- predict(model_glm_lasso, as.matrix(x), type = "response")
y_pred_glmnet <- as.integer(p_pred_glmnet >= 0.5)

table(y_pred_glmnet, y)


