# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-19T17:27:29.768489Z","iopub.execute_input":"2022-02-19T17:27:29.770980Z","iopub.status.idle":"2022-02-19T17:27:34.265253Z"}}
library(tidyverse)
library(data.table)
library(tidymodels)
library(doParallel)
library(themis)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:27:39.337478Z","iopub.execute_input":"2022-02-19T17:27:39.376729Z","iopub.status.idle":"2022-02-19T17:27:39.769357Z"}}
train <- read_csv("../input/titanic/train.csv")
test <- read_csv("../input/titanic/test.csv")
train <- as.data.table(train)
test <- as.data.table(test)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:27:43.310132Z","iopub.execute_input":"2022-02-19T17:27:43.311892Z","iopub.status.idle":"2022-02-19T17:27:43.381725Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:33:05.717631Z","iopub.execute_input":"2022-02-19T17:33:05.719471Z","iopub.status.idle":"2022-02-19T17:33:05.782303Z"}}
train_set <- train[, .(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]
test_set <- test[, .(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]

head(train_set)
head(test_set)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:37:27.435929Z","iopub.execute_input":"2022-02-19T17:37:27.437839Z","iopub.status.idle":"2022-02-19T17:37:27.927019Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:37:44.807447Z","iopub.execute_input":"2022-02-19T17:37:44.809095Z","iopub.status.idle":"2022-02-19T17:39:07.596348Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T17:39:22.782735Z","iopub.execute_input":"2022-02-19T17:39:22.784846Z","iopub.status.idle":"2022-02-19T17:39:22.862284Z"}}
show_best(svm_res, metric = "accuracy", n = 5)

# %% [code]
svm_final <- finalize_model(svm_spec, select_best(svm_res, metric = "accuracy"))
svm_model <- fit(svm_final, Survived ~ ., juice(cust_rec))
pred <- predict(svm_model, test_baked)

# %% [code]
sub <- as.data.table(cbind(test[, .(PassengerId)], pred))
setnames(sub, ".pred_class", "Survived")
write_csv(sub, "submission.csv")