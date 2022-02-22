# %% [code] {"_execution_state":"idle","jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:39.565425Z","iopub.execute_input":"2022-02-22T16:25:39.567041Z","iopub.status.idle":"2022-02-22T16:25:39.582102Z"}}
library(tidyverse)
library(data.table)
library(rsample)
library(torch)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:39.584266Z","iopub.execute_input":"2022-02-22T16:25:39.585519Z","iopub.status.idle":"2022-02-22T16:25:41.624096Z"}}
train <- read_csv("train.csv/train.csv")
test <- read_csv("test.csv/test.csv")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:41.626307Z","iopub.execute_input":"2022-02-22T16:25:41.627571Z","iopub.status.idle":"2022-02-22T16:25:44.776829Z"}}
split <- initial_split(train, 0.8)
split

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:44.779109Z","iopub.execute_input":"2022-02-22T16:25:44.780322Z","iopub.status.idle":"2022-02-22T16:25:44.960385Z"}}
par(mfcol=c(3, 6))
par(mar=c(0, 0, 3, 0), xaxs='i', yaxs='i')
for(i in 1:18) {
  tmp <- matrix(train[i, -1], nrow = 28, ncol = 28)
  tmp <- t(apply(apply(tmp, 2, as.numeric), 1, rev))
  image(1:28, 1:28, tmp, col = gray((0:255)/255), main = paste(train$label[i]), xaxt = "n")
}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:44.964192Z","iopub.execute_input":"2022-02-22T16:25:44.965496Z","iopub.status.idle":"2022-02-22T16:25:44.977988Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:44.980241Z","iopub.execute_input":"2022-02-22T16:25:44.981534Z","iopub.status.idle":"2022-02-22T16:25:45.354871Z"}}
train_set <- mnist_dataset(training(split))
val_set <- mnist_dataset(testing(split))
test_set <- mnist_dataset(test, label = FALSE)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:45.357491Z","iopub.execute_input":"2022-02-22T16:25:45.358818Z","iopub.status.idle":"2022-02-22T16:25:45.382783Z"}}
train_set[1][[1]]$size()
train_set$.getitem(5)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:45.384938Z","iopub.execute_input":"2022-02-22T16:25:45.386115Z","iopub.status.idle":"2022-02-22T16:25:45.600200Z"}}
train_dl <- train_set %>% dataloader(batch_size = 128, shuffle = TRUE)
val_dl <- val_set %>% dataloader(batch_size = 128)
test_dl <- test_set %>% dataloader(batch_size = 128)
train_iter <- train_dl$.iter()
train_iter$.next()
rm(train_iter)

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T16:25:45.602298Z","iopub.execute_input":"2022-02-22T16:25:45.603535Z","iopub.status.idle":"2022-02-22T16:25:45.621248Z"},"jupyter":{"outputs_hidden":false}}
val_loss <- function(model, dl) {
  model$eval()
  l <- c()
  coro::loop(for (b in dl) {
    # get model predictions
    output <- model(b[[1]])
    # get label
    label <- b[[2]]
    # calculate loss
    loss <- nnf_cross_entropy(output, label[,1])
    # track losses
    l <- c(l, loss$item())
  })
  model$train()
  return(l)
}

# %% [code] {"execution":{"iopub.status.busy":"2022-02-22T16:25:45.623684Z","iopub.execute_input":"2022-02-22T16:25:45.624866Z","iopub.status.idle":"2022-02-22T16:25:45.634607Z"},"jupyter":{"outputs_hidden":false}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:45.636672Z","iopub.execute_input":"2022-02-22T16:25:45.637827Z","iopub.status.idle":"2022-02-22T16:25:45.648007Z"}}
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

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:45.650041Z","iopub.execute_input":"2022-02-22T16:25:45.651242Z","iopub.status.idle":"2022-02-22T16:25:45.690205Z"}}
model = net()
optimizer = optim_adam(model$parameters)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2022-02-22T16:25:45.692321Z","iopub.execute_input":"2022-02-22T16:25:45.693537Z","iopub.status.idle":"2022-02-22T16:26:02.772278Z"}}
for (epoch in 1:3) {
  i <- 1
  coro::loop(for (b in train_dl) {
    # make sure each batch's gradient updates are calculated from a fresh start
    optimizer$zero_grad()
    # get model predictions
    output <- model(b[[1]])
    # get label
    label <- b[[2]]
    # calculate loss
    loss <- nnf_cross_entropy(output, label[,1])
    # calculate gradient
    loss$backward()
    # apply weight updates
    optimizer$step()
    # track losses
    i <- i + 1
    if(i %% 50 == 0){
      cat(paste("Epoch:", epoch, "| Batch:", i, "| Loss:", loss$item(), "\n"))
    }
  })
  vl <- val_loss(model, val_dl)
  cat(paste("Epoch:", epoch, "| Val Loss:", mean(vl), "\n"))
}

# %% [code]
preds <- pred(model, test_dl)
preds <- as.data.table(preds)
preds[, ImageId := seq(1, nrow(preds))]
preds <- melt(preds, id.vars = "ImageId", variable.name = "Label")
preds[, highest := max(value), by = "ImageId"]
preds <- preds[value == highest]
preds[, Label := gsub("V", "", Label)]
preds[, Label := as.numeric(Label) - 1]
write_csv(preds[order(ImageId), c(1,2)], "mnist_sub.csv")