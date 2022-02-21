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

xgb_spec <- boost_tree(
  trees = 500,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(), 
  sample_size = tune(), mtry = tune(), learn_rate = tune()
) %>% 
  set_mode("classification") %>%
  set_engine("xgboost", validation = 0.2)

cust_rec <- 
    recipe(Survived ~ ., data = train_set) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = T) %>%
    step_nzv(all_predictors()) %>%
    themis::step_upsample(Survived) %>%
    prep()

test_baked <- bake(cust_rec, new_data = test_set)

folds <- vfold_cv(juice(cust_rec), v = 10 ,strata = Survived)
  
xgb_grid <- grid_latin_hypercube(
    tree_depth(), 
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_set[, -c("Survived")]),
    learn_rate(), 
    size = 25
    )

c <- detectCores()
registerDoParallel(cores = c)
rm(c)
  
ptm <- proc.time()
xgb_res <- tune_grid(
    xgb_spec, 
    Survived ~ .,
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(accuracy)
)
print(proc.time() - ptm)
  
stopImplicitCluster()

show_best(xgb_res, metric = "accuracy", n = 10)

xgb_final <- finalize_model(xgb_spec, select_best(xgb_res, metric = "accuracy"))
xgb_model <- fit(xgb_final, Survived ~ ., juice(cust_rec))
pred <- predict(xgb_model, test_baked)

sub <- as.data.table(cbind(test[, .(PassengerId)], pred))
setnames(sub, ".pred_class", "Survived")
write_csv(sub, "submission.csv")
