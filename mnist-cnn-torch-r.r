library(tidyverse)
library(data.table)
library(rsample)
library(torch)

train <- read_csv("train.csv/train.csv")
test <- read_csv("test.csv/test.csv")

split <- initial_split(train, 0.8)
split

par(mfcol=c(3, 6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for(i in 1:18) {
  tmp <- matrix(train[i, -1], nrow = 28, ncol = 28)
  tmp <- t(apply(apply(tmp, 2, as.numeric), 1, rev))
  image(1:28, 1:28, tmp, col = gray((0:255)/255), main = paste(train$label[i]), xaxt = "n")
}

mnist_dataset <- dataset(
  
  name = "mnist_dataset",
  
  initialize = function(data, label = TRUE) {   
    self$data <- data
    self$label <- label
  },
  
  .getitem = function(index) {
    
    x <- apply(matrix(self$data[index, -1], nrow = 28, ncol = 28), 2, as.numeric)/255
    x <- torch_tensor(t(apply(x, 2, rev)))
    x <- torch_unsqueeze(x, 1)
    
    if(self$label) {
      y <- torch_tensor(self$data$label[index] + 1, dtype = torch_long())
      list(x, y)
    } else {
      list(x)
    }
  },
  
  .length = function() {
    nrow(self$data)
  }  
)

train_set <- mnist_dataset(training(split))
val_set <- mnist_dataset(testing(split))
test_set <- mnist_dataset(test, label = FALSE)

train_set[1][[1]]$size()
train_set$.getitem(5)

train_dl <- train_set %>% dataloader(batch_size = 128, shuffle = TRUE)
val_dl <- val_set %>% dataloader(batch_size = 128)
test_dl <- test_set %>% dataloader(batch_size = 128)
train_iter <- train_dl$.iter()
train_iter$.next()
rm(train_iter)

val_loss <- function(model, dl) {
  model$eval()
  l <- c()
  coro::loop(for (b in dl) {
    output <- model(b[[1]])
    label <- b[[2]]
    loss <- nnf_cross_entropy(output, label[,1])
    l <- c(l, loss$item())
  })
  model$train()
  return(l)
}

pred <- function(model, dl) {
  model$eval()
  preds <- c()
  coro::loop(for (b in dl) {
    with_no_grad({
      batch_pred <- b[[1]] %>% 
        model() %>% 
        nnf_softmax(dim = 2) %>% 
        as_array()
      preds <- rbind(preds, batch_pred)
    })
  })
  return(preds)
}

net <- nn_module(
  "MNIST Net", 
  initialize = function() {
    self$conv1 = nn_conv2d(1, 32, 3, padding = 1)
    self$conv2 = nn_conv2d(32, 96, 3, padding = 1)
    self$conv3x1 = nn_conv2d(96, 16, 1)
    self$fc1 = nn_linear(7*7*16, 256)
    self$fc2 = nn_linear(256, 10)
    self$bnorm1 = nn_batch_norm2d(32)
    self$bnorm2 = nn_batch_norm2d(96)
    self$dropout = nn_dropout2d(0.4)
  }, 
  forward = function(x) {
    x %>%
      self$conv1() %>%
      nnf_relu() %>%
      nnf_max_pool2d(2) %>%
      self$bnorm1() %>%
      self$conv2() %>%
      nnf_relu() %>%
      self$dropout() %>%
      nnf_max_pool2d(2) %>%
      self$bnorm2() %>%
      self$conv3x1() %>%
      nnf_relu() %>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      self$fc2()      
  }
)

model = net()
optimizer = optim_adam(model$parameters)

for (epoch in 1:3) {
  i <- 1
  coro::loop(for (b in train_dl) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    label <- b[[2]]
    loss <- nnf_cross_entropy(output, label[,1])
    loss$backward()
    optimizer$step()
    i <- i + 1
    if(i %% 50 == 0){
      cat(paste("Epoch:", epoch, "| Batch:", i, "| Loss:", loss$item(), "\n"))
    }
  })
  vl <- val_loss(model, val_dl)
  cat(paste("Epoch:", epoch, "| Val Loss:", mean(vl), "\n"))
}

preds <- pred(model, test_dl)
preds <- as.data.table(preds)
preds[, ImageId := seq(1, nrow(preds))]
preds <- melt(preds, id.vars = "ImageId", variable.name = "Label")
preds[, highest := max(value), by = "ImageId"]
preds <- preds[value == highest]
preds[, Label := gsub("V", "", Label)]
preds[, Label := as.numeric(Label) - 1]
write_csv(preds[order(ImageId), c(1,2)], "mnist_sub.csv")
