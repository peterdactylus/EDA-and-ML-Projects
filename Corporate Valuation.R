library(data.table)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(vip)
library(readxl)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

upeer <- read_xlsx("ML_Dataset.xlsx")
upeer <- as.data.table(upeer)
is.na(upeer) <- upeer == "NULL"
upeer <- na.omit(upeer)

upeer[, CreditRating := fcase(CreditRating == "CC", 1,
                              CreditRating == "CCC-", 2,
                              CreditRating == "CCC", 3,
                              CreditRating == "CCC+", 4,
                              CreditRating == "B-", 5,
                              CreditRating == "B", 6,
                              CreditRating == "B+", 7,
                              CreditRating == "BB-", 8,
                              CreditRating == "BB", 9,
                              CreditRating == "BB+", 10,
                              CreditRating == "BBB-", 11,
                              CreditRating == "BBB", 12,
                              CreditRating == "BBB+", 13,
                              CreditRating == "A-", 14,
                              CreditRating == "A", 15,
                              CreditRating == "A+", 16,
                              CreditRating == "AA-", 17,
                              CreditRating == "AA", 18,
                              CreditRating == "AA+", 19,
                              CreditRating == "AAA", 20)]

upeer[, c(4:20) := lapply(.SD, as.numeric), .SDcols = c(4:20)]

prediction <- function(train_set, test_set, wf) {
  folds <- vfold_cv(train_set)
  
  xgb_grid <- grid_latin_hypercube(
    tree_depth(), 
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_set[, -c(1, 12)]),
    learn_rate(),
    stop_iter(c(10, 30)),
    size = 25
  )
  
  c <- detectCores() - 1
  registerDoParallel(cores = c)
  rm(c)
  
  ptm <- proc.time()
  xgb_res <- tune_grid(
    wf,
    resamples = folds,
    grid = xgb_grid,
    metrics = metric_set(rmse)
  )
  print(proc.time() - ptm)
  
  stopImplicitCluster()
  
  #View(show_best(xgb_res, metric = "rmse", n = 10))
  
  xgb_final <- finalize_workflow(wf, select_best(xgb_res, 
                                                 metric = "rmse"))
  
  #print(xgb_final %>% 
          #fit(data = train_set) %>%
          #extract_fit_parsnip() %>%
          #vip(geom = "point"))
  
  xgb_model <- fit(xgb_final, train_set)
  pred_test <- as.data.table(predict(xgb_model, new_data = test_set))
  pred_test[, set := "test"]
  pred_train <- as.data.table(predict(xgb_model, new_data = train_set))
  pred_train[, set := "train"]
  pred <- rbind(pred_train, pred_test)
  
  return(pred)
}

upeer[, ":=" (MktCap = log(MktCap))]

test <- upeer[Identifier %in% c("AG1G.DE"), c(1, 6:9, 14:20)]
train <- upeer[!Identifier %in% c("AG1G.DE"), c(1, 6:9, 14:20)]

xgb_spec <- boost_tree(
  trees = 750,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(), 
  sample_size = tune(), mtry = tune(), learn_rate = tune(), 
  stop_iter = tune()
) %>% 
  set_mode("regression") %>%
  set_engine("xgboost", validation = 0.2)

peer_rec <- 
  recipe(MktCap ~ ., data = train) %>%
  update_role(Identifier, new_role = "ID") %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = T) %>%
  step_nzv(all_predictors())

wf <- workflow(peer_rec, xgb_spec)

pred <- prediction(train, test, wf)

result <- data.table()
for(i in 1:50) {
  pred <- prediction(train, test, wf)
  pred <- as.data.table(pred)
  result <- rbind(result, pred[set == "test"])
  
  print(i)
}

result[, MktCap_pred := exp(.pred)]
result[, Forecast := rep(c("Broker", "Own"), 50)]
ref <- data.table(Valuation = c("MktCap", "DCF"), 
                  c(upeer[Identifier == "AG1G.DE", exp(unique(MktCap))], 
                    8494000000))

ggplot() +
  geom_density(data = result, aes(MktCap_pred, fill = Forecast), 
               adjust = 1.5, size = 0, alpha = 0.6, color = "white") +
  geom_vline(data = ref, aes(xintercept = V2, color = Valuation), size = 1) +
  scale_color_manual(values = c("MktCap" = "orange", "DCF" = "purple")) +
  labs(x = "Valuation", 
       title = "Distribution of Machine Learning Predictions based on Forecasts for 2025",
       subtitle = "Own Forecasts are more pessimistic in the short but more optimistic in the long term") +
  expand_limits(x = c(3e+09, 1.15e+10)) +
  theme_bw() +
  theme(text = element_text(family = "serif", size = 13)) +
  theme(plot.title = element_text(size = 12, face = "bold")) +
  theme(plot.subtitle = element_text(size = 11))
