cv.glm.custom = function (data, glmfit, cost = function(y, yhat) mean((y - yhat)^2), 
          K = n) 
{ 
  #Custom version - Author: K Anvesh, Version:1.1, Date:1/16/2016"
  
  library(ROCR)
  call <- match.call()
  if (!exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) 
    runif(1)
  seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  n <- nrow(data)
  if ((K > n) || (K <= 1)) 
    stop("'K' outside allowable range")
  K.o <- K
  K <- round(K)
  kvals <- unique(round(n/(1L:floor(n/2))))
  temp <- abs(kvals - K)
  if (!any(temp == 0)) 
    K <- kvals[temp == min(temp)][1L]
  if (K != K.o) 
    warning(gettextf("'K' has been set to %f", K), domain = NA)
  f <- ceiling(n/K)
  s <- sample(rep(1L:K, f), n)
  n.s <- table(s)
  glm.y <- glmfit$y
  cost.0 <- cost(glm.y, fitted(glmfit))
  ms <- max(s)
  CV <- 0
  Call <- glmfit$call
  auc <- 0
  for (i in seq_len(ms)) {
    j.out <- seq_len(n)[(s == i)]
    j.in <- seq_len(n)[(s != i)]
    Call$data <- data[j.in, , drop = FALSE]
    d.glm <- eval.parent(Call)
    p.alpha <- n.s[i]/n
    cost.i <- cost(glm.y[j.out], predict(d.glm, data[j.out, 
                                                     , drop = FALSE], type = "response"))
    CV <- CV + p.alpha * cost.i
    cost.0 <- cost.0 - p.alpha * cost(glm.y, predict(d.glm, 
                                                     data, type = "response"))
    auc = auc +  p.alpha *slot(performance(prediction(predict(d.glm, 
                                                    data, type = "response"),glm.y), "auc"), "y.values")[[1]]
  }
  list(call = call, K = K, delta = as.numeric(c(CV, CV + cost.0)), 
       seed = seed, auc=auc)
}
