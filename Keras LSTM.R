library(keras)
library(data.table)
library(tidyverse)
library(tidymodels)
library(readxl)
library(GGally)

coins <- read_xlsx("crypto_lstm.xlsx", sheet = 2)
coins <- as.data.table(coins)

ggcorr(coins, label = T, label_round = 2)

eth <- coins$ETH
out <- c()
for(i in 15:(length(eth) - 2)) {
  for(j in 0:2) {
    out <- append(out, prod(eth[i:(i + j)]))
  }
}
rm(i, j, eth)

out <- as.data.table(out)
i <- c(1:(nrow(out) / 3))
out[, period := rep(i, each = 3)]
out[, res := fifelse(out >= 1.05, "rise", fifelse(out <= 0.95, "fall", "stay"))]
out <- out[, .N, by = c("period", "res")]
out <- dcast(out, ... ~ res, value.var = "N")
out[, ":=" (rise = fifelse(rise > 0 & is.na(fall), 1, 0, na = 0), 
            fall = fifelse(fall > 0 & is.na(rise), 1, 0, na = 0),
            stay = fifelse(stay == 3 | (rise > 0 & fall > 0), 1, 0, na = 0))]

set <- data.table()
for(i in 1:ncol(coins)) {
  coin <- coins[, ..i]
  tmp <- c()
  for(j in 1:(nrow(coin) - 17)) {
    for(k in 0:13) {
      tmp <- append(tmp, prod(coin[j:(j + k)]))
    }
  }
  set <- cbind(set, tmp)
}
rm(i, j, k, tmp, coin)

colnames(set) <- colnames(coins)

train_x <- as.matrix(set[1:(nrow(set) - 50 * 14)])
test_x <- as.matrix(set[(nrow(set) - 50 * 14 + 1):nrow(set)])
train_x <- aperm(array(train_x, dim = c(14, nrow(train_x) / 14, 7)), 
               c(2, 1, 3))
test_x <- aperm(array(test_x, dim = c(14, nrow(test_x) / 14, 7)), 
                c(2, 1, 3))

train_y <- as.matrix(out[1:(nrow(out) - 50), -c("period")])
test_y <- as.matrix(out[(nrow(out) - 49):nrow(out), -c("period")])

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

model <- keras_model_sequential() %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2, 
             activation = "relu", input_shape = c(14, 7)) %>%
  layer_dense(units = 96, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 3, activation = "softmax")

model %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam", 
  metrics = "accuracy"
)

history <- model %>% fit(
  test_x,
  test_y,
  batch_size = 16,
  epochs = 50,
  shuffle = TRUE,
  validation_split = 0.2,
  callbacks = early_stop
)
