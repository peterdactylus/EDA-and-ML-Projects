library(tidyverse)
library(data.table)
library(tidymodels)
library(doParallel)
library(themis)

train <- read_csv("../input/titanic/train.csv")
test <- read_csv("../input/titanic/test.csv")
train <- as.data.table(train)
test <- as.data.table(test)

train[grep("Mr.", train$Name), Title := "Mr."]
train[grep("Mrs.", train$Name), Title := "Mrs."]
train[grep("Miss.", train$Name), Title := "Miss."]

test[grep("Mr.", test$Name), Title := "Mr."]
test[grep("Mrs.", test$Name), Title := "Mrs."]
test[grep("Miss.", test$Name), Title := "Miss."]

train[, Pclass := as.character(Pclass)]
test[, Pclass := as.character(Pclass)]

train[, Survived := as.character(Survived)]

head(train)

train_set <- train[, .(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]
test_set <- test[, .(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]

head(train_set)
head(test_set)

svm_spec <- svm_rbf(
    cost = tune(), 
    rbf_sigma = tune()
) %>%
    set_mode("classification") %>%
    set_engine("kernlab")

cust_rec <- 
    recipe(Survived ~ ., data = train_set) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = T) %>%
    step_string2factor(all_nominal_predictors()) %>%
    step_impute_knn(all_predictors(), neighbors = 10) %>%
    step_nzv(all_predictors()) %>%
    themis::step_upsample(Survived) %>%
    prep()

test_baked <- bake(cust_rec, new_data = test_set)

folds <- vfold_cv(juice(cust_rec), v = 10 ,strata = Survived)
  
svm_grid <- grid_latin_hypercube(
    cost(), 
    rbf_sigma(), 
    size = 15
)

c <- detectCores()
registerDoParallel(cores = c)
rm(c)
  
ptm <- proc.time()
svm_res <- tune_grid(
    svm_spec, 
    Survived ~ .,
    resamples = folds,
    grid = svm_grid,
    metrics = metric_set(accuracy)
)
print(proc.time() - ptm)
  
stopImplicitCluster()

show_best(svm_res, metric = "accuracy", n = 5)

svm_final <- finalize_model(svm_spec, select_best(svm_res, metric = "accuracy"))
svm_model <- fit(svm_final, Survived ~ ., juice(cust_rec))
pred <- predict(svm_model, test_baked)

sub <- as.data.table(cbind(test[, .(PassengerId)], pred))
setnames(sub, ".pred_class", "Survived")
write_csv(sub, "submission.csv")
