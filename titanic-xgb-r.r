# %% [code] {"_execution_state":"idle","execution":{"iopub.status.busy":"2022-02-19T16:38:03.077381Z","iopub.execute_input":"2022-02-19T16:38:03.079639Z","iopub.status.idle":"2022-02-19T16:38:06.674316Z"}}
library(tidyverse)
library(data.table)
library(tidymodels)
library(doParallel)
library(themis)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:38:11.421707Z","iopub.execute_input":"2022-02-19T16:38:11.451742Z","iopub.status.idle":"2022-02-19T16:38:11.77579Z"}}
train <- read_csv("../input/titanic/train.csv")
test <- read_csv("../input/titanic/test.csv")
train <- as.data.table(train)
test <- as.data.table(test)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:38:14.665613Z","iopub.execute_input":"2022-02-19T16:38:14.667101Z","iopub.status.idle":"2022-02-19T16:38:14.728668Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:38:20.3563Z","iopub.execute_input":"2022-02-19T16:38:20.357876Z","iopub.status.idle":"2022-02-19T16:38:20.389956Z"}}
train_set <- train[, .(Survived, Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]
test_set <- test[, .(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Title)]

head(train_set)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:39:11.338226Z","iopub.execute_input":"2022-02-19T16:39:11.339791Z","iopub.status.idle":"2022-02-19T16:39:11.60745Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:39:19.892497Z","iopub.execute_input":"2022-02-19T16:39:19.894169Z","iopub.status.idle":"2022-02-19T16:40:29.953503Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T16:40:42.186551Z","iopub.execute_input":"2022-02-19T16:40:42.188452Z","iopub.status.idle":"2022-02-19T16:40:42.26279Z"}}
show_best(xgb_res, metric = "accuracy", n = 10)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T14:39:44.606793Z","iopub.execute_input":"2022-02-19T14:39:44.609156Z","iopub.status.idle":"2022-02-19T14:39:45.532612Z"}}
xgb_final <- finalize_model(xgb_spec, select_best(xgb_res, metric = "accuracy"))
xgb_model <- fit(xgb_final, Survived ~ ., juice(cust_rec))
pred <- predict(xgb_model, test_baked)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-19T14:53:40.760876Z","iopub.execute_input":"2022-02-19T14:53:40.762526Z","iopub.status.idle":"2022-02-19T14:53:40.79876Z"}}
sub <- as.data.table(cbind(test[, .(PassengerId)], pred))
setnames(sub, ".pred_class", "Survived")
write_csv(sub, "submission.csv")