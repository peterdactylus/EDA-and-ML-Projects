library(tidyverse)
library(data.table)
library(tidymodels)
library(doParallel)
library(stringr)
library(vip)

train <- read_csv("train.csv")
test <- read_csv("test.csv")
train <- as.data.table(train)
test <- as.data.table(test)

head(train)
train[, .N, by = "Transported"]

#PassengerId
train[, PassengerClass := str_sub(PassengerId, start = -2)]
test[, PassengerClass := str_sub(PassengerId, start = -2)]

ggplot(train, aes(Transported, fill = PassengerClass)) +
  geom_bar(position = "fill") +
  theme_minimal()

#HomePlanet
ggplot(train, aes(HomePlanet, fill = Transported)) +
  geom_bar() +
  theme_minimal()

#CryoSleep
ggplot(train, aes(CryoSleep, fill = Transported)) +
  geom_bar() +
  theme_minimal()

#Cabin
train[, CabinFirst := str_sub(Cabin, end = 1)]
train[, CabinLast := str_sub(Cabin, start = -1)]
test[, CabinFirst := str_sub(Cabin, end = 1)]
test[, CabinLast := str_sub(Cabin, start = -1)]

ggplot(train, aes(CabinFirst, fill = Transported)) +
  geom_bar() +
  theme_minimal()

ggplot(train, aes(CabinLast, fill = Transported)) +
  geom_bar() +
  theme_minimal()

ggplot(train[!is.na(Cabin)], aes(CabinLast, fill = CabinFirst)) +
  geom_bar() +
  theme_minimal()

#Destination
ggplot(train, aes(Destination, fill = Transported)) +
  geom_bar() +
  theme_minimal()

#Age
ggplot(train[!is.na(Age)], aes(Age)) +
  geom_density() +
  theme_minimal()

shapiro.test(sample(train[!is.na(Age), Age], size = 5000, replace = T))

ggplot(train[!is.na(Age)], aes(Transported, Age)) +
  geom_boxplot() +
  theme_minimal()

wilcox.test(Age ~ Transported, data = train[!is.na(Age)])

#VIP
ggplot(train, aes(VIP, fill = Transported)) +
  geom_bar() +
  theme_bw()

#Total Expenditures
train[, Expenditures := RoomService + FoodCourt + ShoppingMall + Spa + VRDeck]
test[, Expenditures := RoomService + FoodCourt + ShoppingMall + Spa + VRDeck]

ggplot(train[!is.na(Expenditures), .(Expenditures = Expenditures + 1, Transported)], 
       aes(Transported, Expenditures)) +
  geom_boxplot() +
  scale_y_log10() +
  theme_bw()

train[Expenditures > 0, .N, by = "Transported"]

#RoomService
ggplot(train[!is.na(RoomService), .(RoomService = RoomService + 1, Transported)], 
       aes(RoomService, fill = Transported, color = Transported)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  theme_bw()

train[RoomService > 0, .N, by = "Transported"]

#FoodCourt
ggplot(train[!is.na(FoodCourt), .(FoodCourt = FoodCourt + 1, Transported)], 
       aes(FoodCourt, fill = Transported, color = Transported)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  theme_bw()

train[FoodCourt > 0, .N, by = "Transported"]

#ShoppingMall
ggplot(train[!is.na(ShoppingMall), .(ShoppingMall = ShoppingMall + 1, Transported)], 
       aes(ShoppingMall, fill = Transported, color = Transported)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  theme_bw()

train[ShoppingMall > 0, .N, by = "Transported"]

#Spa
ggplot(train[!is.na(Spa), .(Spa = Spa + 1, Transported)], 
       aes(Spa, fill = Transported, color = Transported)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  theme_bw()

train[Spa > 0, .N, by = "Transported"]

#VRDeck
ggplot(train[!is.na(VRDeck), .(VRDeck = VRDeck + 1, Transported)], 
       aes(VRDeck, fill = Transported, color = Transported)) +
  geom_density(alpha = 0.5) +
  scale_x_log10() +
  theme_bw()

train[VRDeck > 0, .N, by = "Transported"]

#Model
train_set <- train[, c(2:3, 5:6, 8:12, 14:18)]
train_set[, Transported := fifelse((Transported), "1", "0")]
train_set[, CryoSleep := fifelse((CryoSleep), "1", "0")]
test_set <- test[, c(2:3, 5:6, 8:12, 14:17)]
test_set[, CryoSleep := fifelse((CryoSleep), "1", "0")]

xgb_rec <- 
  recipe(Transported ~ ., data = train_set) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors(), neighbors = 10) %>%
  step_other(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = T) %>%
  step_nzv(all_predictors())

rf_rec <- 
  recipe(Transported ~ ., data = train_set) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_knn(all_predictors(), neighbors = 10) %>%
  step_other(all_nominal_predictors())

prep(xgb_rec)
prep(rf_rec)

xgb_spec <- boost_tree(
  trees = 2000,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(), 
  sample_size = tune(), mtry = tune(), learn_rate = tune(), 
  stop_iter = tune()
) %>% 
  set_mode("classification") %>%
  set_engine("xgboost", validation = 0.2)

rf_spec <- rand_forest(
  trees = 1000, 
  mtry = tune(), 
  min_n = tune()
) %>%
  set_mode("classification") %>% 
  set_engine("ranger", importance = "permutation")

xgb_wf <- workflow(xgb_rec, xgb_spec)
rf_wf <- workflow(rf_rec, rf_spec)

folds <- vfold_cv(train_set, v = 5)

xgb_grid <- grid_latin_hypercube(
  tree_depth(), 
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), juice(prep(xgb_rec))),
  learn_rate(),
  stop_iter(c(20, 50)),
  size = 25
)

rf_grid <- grid_latin_hypercube(
  finalize(mtry(), juice(prep(rf_rec))), 
  min_n(), 
  size = 15
)

c <- detectCores() - 1
registerDoParallel(cores = c)
rm(c)

ptm <- proc.time()
xgb_res <- tune_grid(
  xgb_wf,
  resamples = folds,
  grid = xgb_grid,
  metrics = metric_set(accuracy)
)
print(proc.time() - ptm)

ptm <- proc.time()
rf_res <- tune_grid(
  rf_wf,
  resamples = folds,
  grid = rf_grid,
  metrics = metric_set(accuracy)
)
print(proc.time() - ptm)

stopImplicitCluster()

View(show_best(xgb_res, metric = "accuracy", n = 10))
View(show_best(rf_res, metric = "accuracy"))

xgb_final <- finalize_workflow(xgb_wf, 
                               select_best(xgb_res, metric = "accuracy"))
rf_final <- finalize_workflow(rf_wf, 
                              select_best(rf_res, metric = "accuracy"))

xgb_final %>% 
  fit(data = train_set) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point", num_features = 15)

rf_final %>% 
  fit(data = train_set) %>%
  extract_fit_parsnip() %>%
  vip(geom = "point", num_features = 10)

xgb_model <- fit(xgb_final, train_set)
xgb_pred <- predict(xgb_model, new_data = test_set, type = "prob")

rf_model <- fit(rf_final, train_set)
rf_pred <- predict(rf_model, new_data = test_set, type = "prob")

sub <- data.table(PassengerId = test$PassengerId, 
                  Transported = 0.5 * xgb_pred$.pred_1 + 0.5 * rf_pred$.pred_1)
sub[, Transported := fifelse(Transported > 0.5, "True", "False")]
write_csv(sub, "submission.csv")
