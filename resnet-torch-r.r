net <- nn_module(
    "ResNet", 
    initialize = function() {
        self$conv1 = nn_conv2d(1, 32, 3, padding = 1)
        self$conv2r1 = nn_conv2d(32, 32, 3, padding = 1)
        self$conv3r1 = nn_conv2d(32, 32, 3, padding = 1)
        self$conv4x1 = nn_conv2d(32, 64, 1)
        self$conv5r2 = nn_conv2d(64, 64, 3, padding = 1)
        self$conv6r2 = nn_conv2d(64, 64, 3, padding = 1)
        self$conv7x1 = nn_conv2d(64, 16, 1)
        self$fc1 = nn_linear(7*7*16, 256)
        self$fc2 = nn_linear(256, 10)
        self$bnorm1 = nn_batch_norm2d(32)
        self$bnorm2 = nn_batch_norm2d(64)
    }, 
    forward = function(x) {
        x = self$conv1(x)
        x = nnf_relu(x)
        x = nnf_max_pool2d(x, 2)
        x = self$bnorm1(x)
        y = self$conv2r1(x)
        y = nnf_relu(y)
        y = self$conv3r1(y)
        x = y + x
        x = nnf_relu(x)
        x = self$conv4x1(x)
        x = nnf_max_pool2d(x, 2)
        x = self$bnorm2(x)
        y = self$conv5r2(x)
        y = nnf_relu(y)
        y = self$conv6r2(y)
        x = y + x
        x = nnf_relu(x)
        x = self$conv7x1(x)
        x = torch_flatten(x, start_dim = 2)
        x = self$fc1(x)
        x = nnf_relu(x)
        x = self$fc2(x)      
    }
)