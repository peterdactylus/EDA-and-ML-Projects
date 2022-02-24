library(tidyverse)
library(tidymodels)
library(data.table)
library(doParallel)
library(vip)
library(themis)
library(lubridate)
library(OneR)

set.seed(123)

setwd(dirname(rstudioapi::getSourceEditorContext()$path))

customers <- as.data.table(read_csv("customers.csv"))
geo <- as.data.table(read_csv("geo.csv"))
transactions <- as.data.table(read_csv("transactions.csv"))

customers[, COUNTRY := fifelse(COUNTRY == "France", "FR", "CH")]
data <- merge(transactions, geo, by = "SALES_LOCATION")
data[, CUSTOMER := gsub('"', '', CUSTOMER)]
customers[, ":=" (CUSTOMER = as.character(CUSTOMER), 
                  REV_CURRENT_YEAR = gsub('"', '', REV_CURRENT_YEAR))]
customers[, REV_CURRENT_YEAR := as.numeric(REV_CURRENT_YEAR)]
data <- merge(data, customers, all.x = T, by = c("CUSTOMER", "COUNTRY"))
data[, OFFER_STATUS := substr(OFFER_STATUS, 1, 1)]
data[, OFFER_STATUS := fifelse(OFFER_STATUS == "W", "1", "0")]

data[, Contact_Year := dmy(CREATION_YEAR)]
data[, Contact_Year := year(Contact_Year)]
data[, ReProd_Costs := COSTS_PRODUCT_A + COSTS_PRODUCT_B + 
       COSTS_PRODUCT_C + COSTS_PRODUCT_D + COSTS_PRODUCT_E]

woe_buckets <- function(vname) {
  tmp1 <- data[OFFER_STATUS == "1", .N / nrow(data[OFFER_STATUS == "1"]), 
               by = vname]
  tmp0 <- data[OFFER_STATUS == "0", .N / nrow(data[OFFER_STATUS == "0"]), 
               by = vname]
  tmp <- merge(tmp1, tmp0, by = vname, all = T)
  tmp[is.na(V1.x), V1.x := 1 / nrow(data[is.na(TEST_SET_ID) & 
                                           OFFER_STATUS == "1"])]
  tmp[is.na(V1.y), V1.y := 1 / nrow(data[is.na(TEST_SET_ID) & 
                                           OFFER_STATUS == "0"])]
  tmp[, woe := log(V1.x / V1.y)]
  tmp <- merge(data[, ..vname], tmp, by = vname, all = T)
  tmp[, bins := bin(woe, nbins = 5, method = "content", na.omit = F)]
  tmp <- unique(tmp, by = vname)
  tmp[, c("V1.x", "V1.y") := lapply(.SD, sum), 
      .SDcols = c("V1.x", "V1.y"), by = "bins"]
  tmp[, woe := log(V1.x / V1.y), by = "bins"] %>% 
    .[, iv := woe * (V1.x - V1.y), by = "bins"]
  setnames(tmp, "bins", paste0(vname, "bins"))
  
  data <<- merge(data, tmp[, c(1, 5)], by = vname, all.x = T)
}

vnames <- colnames(data[, c(3, 12, 16, 25)])
lapply(vnames, function(x) {woe_buckets(x)})

prediction <- function(train_set, test_set, wf) {
  folds <- vfold_cv(train_set, v = 5 ,strata = OFFER_STATUS)
  
  xgb_grid <- grid_latin_hypercube(
    tree_depth(), 
    min_n(),
    loss_reduction(),
    sample_size = sample_prop(),
    finalize(mtry(), train_set[, -c("OFFER_STATUS")]),
    learn_rate(),
    stop_iter(c(15,40)),
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
    metrics = metric_set(bal_accuracy)
  )
  print(proc.time() - ptm)
  
  stopImplicitCluster()
  
  #View(show_best(xgb_res, metric = "bal_accuracy", n = 10))
  
  xgb_final <- finalize_workflow(wf, select_best(xgb_res, 
                                                 metric = "bal_accuracy"))
  
  #print(xgb_final %>% 
  #        fit(data = train_set) %>%
  #        extract_fit_parsnip() %>%
  #        vip(geom = "point", num_features = 15))
  
  xgb_model <- fit(xgb_final, train_set)
  pred <- as.data.table(predict(xgb_model, new_data = test_set, 
                                type = "raw"))
  
  return(pred)
}

train <- data[is.na(TEST_SET_ID), c(10:13, 17:18, 22, 26:27, 31, 33:38)]
test <- data[!is.na(TEST_SET_ID), c(10:13, 17:18, 22, 26:27, 31, 33:38)]

xgb_spec <- boost_tree(
  trees = 2000,
  tree_depth = tune(), min_n = tune(), loss_reduction = tune(), 
  sample_size = tune(), mtry = tune(), learn_rate = tune(), 
  stop_iter = tune()
) %>% 
  set_mode("classification") %>%
  set_engine("xgboost", validation = 0.2)

cust_rec <- 
  recipe(OFFER_STATUS ~ ., data = train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_other(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = T) %>%
  step_nzv(all_predictors()) %>%
  themis::step_downsample(OFFER_STATUS)

wf <- workflow(cust_rec, xgb_spec)

sub <- data.table()
for(i in 1:3) {
  set.seed(i)
  pred <- prediction(train, test, wf)
  setnames(pred, "V1", paste0("pred", i))
  sub <- cbind(sub, pred)
}

sub <- cbind(data[!is.na(TEST_SET_ID), c(25)], sub)
sub[, prediction := (pred1 + pred2 + pred3) / 3]
sub[, prediction := fifelse(prediction <= 0.5, 1, 0)]

setnames(sub, c("TEST_SET_ID"), c("id"))
write_csv(sub[order(id), -c(2:4)], "submission.csv")
write_csv(sub, "interim_sub_adj.csv")
