set.seed(40)
Value <- function(data, children=list(), op = "", name = NULL, 
                  grad_fn = function(){}) {
  structure(
    rlang::env(
      data = data,
      children = children,
      op = op,
      id = uuid::UUIDgenerate(),
      name = name,
      grad = 0,           # Por padrão iniciamos como sendo zero
      grad_fn = grad_fn   
    ), 
    class = "Value"
  )
}

print.Value <- function(x) {
  cat("<Value data=", x$data, ">\n")
}

`+.Value` <- function(x, y) { 
  out <- Value(
    x$data + y$data, 
    children = list(x, y), 
    op = "+",
    name = paste(x$name, "+", y$name)
  ) 
  
  out$grad_fn <- function() {
    x$grad <- x$grad + 1 * out$grad
    y$grad <- y$grad + 1 * out$grad
  }
  
  out
}


`-.Value` <- function(x, y) { 
  out <- Value(
    x$data - y$data, 
    children = list(x, y), 
    op = "-",
    name = paste(x$name, "-", y$name)
  ) 
  
  out$grad_fn <- function() {
    x$grad <- x$grad + 1 * out$grad
    y$grad <- y$grad - 1 * out$grad
  }
  
  out
}


`*.Value` <- function(x, y) { 
  out <- Value(
    x$data * y$data, 
    children = list(x, y), 
    op = "*",
    name = paste(x$name, "*", y$name)
  ) 
  
  out$grad_fn <- function() {
    x$grad <- x$grad + y$data * out$grad
    y$grad <- y$grad + x$data * out$grad
  }
  
  out
}


`^.Value` <- function(x, y) { 
  out <- Value(
    x$data ^ y,                      # Vamos assumir que o "y" não é um Value
    # Isso porque, quando temos uma uma função do tipo x^y, a derivada é 
    # d/dy (x^y) = (x^y)lnx
    children = list(x), 
    op = "^",
    name = paste(x$name, "^", y)
  ) 
  
  out$grad_fn <- function() {
    x$grad <- x$grad + y * (x$data^(y - 1)) * out$grad
  }
  
  out
}


tanh.Value <- function(x) {
  
  t <- tanh(x$data)
  
  out <- Value(
    t,
    children = list(x),
    op = "tanh",
    name = paste("tanh(", x$name, ")")
  )
  
  out$grad_fn <- function() {
    x$grad <- x$grad + ((1 - t^2)) * out$grad
  }
  
  out
}

# Função Sigmóide 
sigm.Value <- function(x){
  
  sigmoide <- 1 / (1 + exp(-x$data))
  
  out <- Value(
    sigmoide,
    children = list(x),
    op = "sigmoide",
    name = paste("sigmoide(", x$name, ")")
  )
  out$grad_fn <- function()
  {
    x$grad <- x$grad + (sigmoide*(1 - sigmoide)) * out$grad
  }
  out
}

# Funções auxiliares
topo_sort <- function(node, topo = list(), visited = c()) {
  if (!inherits(node, "Value")) browser()
  if (node$id %in% visited) return(list(topo, visited))
  visited <- c(visited, node$id)
  for (child in node$children) {
    .[topo, visited] <- topo_sort(child, topo, visited)
  }
  topo <- c(topo, node)
  list(topo, visited)
}

backprop <- function(value, visited = c()) {
  value$grad <- 1
  .[topo, ..] <- topo_sort(value)
  for (node in rev(topo)) {
    node$grad_fn()
  }
}

sigm <- function(x) sigm.Value(x)

Neuron <- R6::R6Class(
  lock_objects = FALSE,      # Senão, não deixa colocar/adicionar objetos no "self". 
  public = list(
    initialize = function(nin = 2) {
      self$ws <- lapply(seq_len(nin), function(i) {
        Value(rnorm(1), name = paste0("w", i))
      })
      self$b <- Value(1, name = "b")
    },
    forward = function(x) {
      z <- self$b
      for (i in seq_along(x)) {
        z <- z + x[[i]] * self$ws[[i]]
      }
      z
    }
  )
)

# Camada única
Layer <- R6::R6Class(
  lock_objects = FALSE, 
  public = list(
    initialize = function(nin, nout, act) {
      self$neurons <- lapply(seq_len(nout), function(i) {
        Neuron$new(nin = nin)
      })
      self$act <- act
    }, 
    forward = function(x) {                 # Preciso retornar uma Lista, que para cada 
      lapply(self$neurons, function(n) {   # Neurônio tenha "act(n$forward(x))"
        self$act(n$forward(x))
      })                                 
    }
  )
)

# Multilayer Perceptron (MLP)
MLP <- R6::R6Class(
  lock_objects = FALSE, 
  public = list(
    initialize = function(nin, nouts = arquitetura_rede) {
      self$layers <- list()
      for (i in seq_along(nouts)) {
        self$layers[[i]] <- Layer$new(nin, nouts[i], sigm)
        nin <- nouts[i]
      }
    },
    forward = function(x){
      o <- x                         
      for (l in self$layers) {       # Dessa forma, aplicamos as camadas em
        o <- l$forward(o)            # sequência. Aplico a primeira, depois 
      }                              # mudo o "o".
      o
    }
  )
)

# Dados de entrada no formato (x1, x2)
x <- list(
  c(1, 1), 
  c(1, 0),
  c(0, 1),
  c(0, 0)
)

xs <- lapply(seq_along(x), function(i) {
  lapply(seq_along(x[[i]]), function(j) Value(x[[i]][[j]], 
                                              name = paste0("x", i, j)))
})

# Saída esperada
d <- c(1, 0, 0, 0)
ys <- lapply(seq_along(d), function(i) Value(d[i], name = paste0("y", i)))

# o número de camadas ocultas

nro_camadas_ocultas <- 1:1

# o número de neurônios em cada camada
nro_neuron_cada_camada <- 2

# durante quantas épocas realizará o treinamento
epochs <- 500

loss_history <- numeric(epochs)
eta_value <- 0.1

arquitetura_rede <- rep(nro_neuron_cada_camada, length(nro_camadas_ocultas))
arquitetura_rede <- c(arquitetura_rede, 1)

net <- MLP$new(nin = 2, nouts = arquitetura_rede)

inicio <- Sys.time()
for (i in 1:epochs) {
  
  os <- lapply(xs, net$forward)
  
  L <- (os[[1]][[1]] - ys[[1]])^2 +
    (os[[2]][[1]] - ys[[2]])^2 +
    (os[[3]][[1]] - ys[[3]])^2 + 
    (os[[4]][[1]] - ys[[4]])^2
  
  backprop(L)
  
  for (layer in net$layers) {
    for(neuron in layer$neurons) {
      for(w in neuron$ws) { 
        eta <- eta_value
        w$data <- w$data - eta * w$grad
        w$grad <- 0                      # Uma vez que usou o grad, depois zera ele.
      }
      neuron$b$data <- neuron$b$data - eta * neuron$b$grad
      neuron$b$grad <- 0
    }
  }
  
  loss_history[i] <- L$data
}
fim <- Sys.time()

library(ggplot2)
df <- data.frame(
  epoch = 1:epochs,
  loss = loss_history
)

ggplot(df, aes(x = epoch, y = loss)) +
  geom_line(color = "blue", size = 1) +
  
  theme_minimal() +
  labs(title = "Evolução da Loss ao longo das Épocas",
       x = "Época",
       y = "Loss")

tempo_total <- fim - inicio
print(tempo_total)

y_pred <- sapply(os, function(o) ifelse(o[[1]]$data > 0.5, 1, 0))
print(y_pred)