library(nleqslv)
library(MASS)
library(randomForest)
library(mgcv)
set.seed(2024)
sigma <- matrix(c(1,-0.25,-0.25,-0.25,1,-0.25,-0.25,-0.25,1),3,3)
n2 <- 10^6
c2 <-  mvrnorm(n2,c(1,-1,0),sigma)
x1 <- c2[,1]
x2 <- c2[,2]
x3 <- c2[,3] #shadow
prob_y1 <- 1 / (1+exp(x1+x2+x3))
y1 <- sapply(prob_y1,function(x) rbinom(1,1,x))
d_opt <- (x1 + x2 + x3 < 0)*1
d1 <- rep(0.5,n2)
d2 <- d_opt * 0.4 + d1 * 0.6
d3 <- d_opt * 0.7 + d1 * 0.3
Value_true_d1 <- mean(d1*y1)
Value_true_d2 <- mean(d2*y1)
Value_true_d3 <- mean(d3*y1)

etaFun_eff <- function(eta, cov_matrix, cov_matrix_1, cov_matrix_0, moments, trt, prob){ 
  ps <- as.vector(1 / (1+exp(-cov_matrix%*%eta)))
  ps_1 <- as.vector(1 / (1+exp(-cov_matrix_1%*%eta)))
  ps_0 <- as.vector(1 / (1+exp(-cov_matrix_0%*%eta)))
  moments_eff <- ((1-ps_1)/ps_1*prob + (1-ps_0)/ps_0*(1-prob)) / ((ps_1-1)/ps_1/ps_1*prob + (ps_0-1)/ps_0/ps_0*(1-prob)) * moments
  colSums((1-trt/ps)*moments_eff)
}

etaFun <- function(eta, cov_matrix, moments, trt){ 
  ps <- as.vector(1 / (1+exp(-cov_matrix%*%eta)))
  colSums((1-trt/ps)*moments)
}

library(foreach)
library(doParallel)
no_cores <- detectCores(logical=F)
cl <- makeCluster(no_cores)
registerDoParallel(cl)

sim <- foreach(k=1:500,.combine='cbind',.packages = c("MASS","randomForest","nleqslv","mgcv"))%dopar% {
  set.seed(k)
  n1 <- 1000
  C1 <- matrix(0,n1,3)
  colnames(C1) <- c("X1","X2","X3")
  C1 <- mvrnorm(n1,c(1,-1,0),sigma)
  X1 <- C1[,1] 
  X2 <- C1[,2]
  X3 <- C1[,3] #shadow variable
  prob_Y1 <- 1 / (1+exp(X1+X2+X3))
  Y1 <- sapply(prob_Y1,function(x) rbinom(1,1,x))
  ps1 <- 1/(1+exp(-(X1-0.5*X2+0.7*Y1)))
  A1 <- sapply(ps1,function(x) rbinom(1,1,x))
  dat <- data.frame(Y1,C1,A1)
  
  index_trt1 <- which(A1==1)
  dat_trt1 <- dat[index_trt1,]
  fit_om <- gam(Y1 ~ X1 + X2 + X3, data = dat_trt1, family = "binomial")
  prob_Y1_pred <- as.vector(predict(fit_om,dat,type = "response"))
  
  CM1 <- cbind(rep(1,n1),X1,X2,Y1)
  CM1_1 <- cbind(rep(1,n1),X1,X2,rep(1,n1))
  CM1_0 <- cbind(rep(1,n1),X1,X2,rep(0,n1))
  M1 <- cbind(rep(1,n1),C1)

  eta.hat_initial <- nleqslv(x=c(0,1,-0.5,.1),fn=etaFun,cov_matrix=CM1,moments=M1,trt=A1)$x
  ps.hat_initial <- as.vector(1 / (1+exp(-CM1%*%eta.hat_initial)))
  eta.hat <- nleqslv(x=c(0,1,-0.5,.1),fn=etaFun_eff,cov_matrix=CM1,cov_matrix_1=CM1_1,cov_matrix_0=CM1_0,moments=M1,trt=A1,prob=prob_Y1_pred)$x
  ps.hat <- as.vector(1 / (1+exp(-CM1%*%eta.hat)))
  ps.hat_1 <- as.vector(1 / (1+exp(-CM1_1%*%eta.hat)))
  ps.hat_0 <- as.vector(1 / (1+exp(-CM1_0%*%eta.hat)))
  
  num <- (1-ps.hat_1)/ps.hat_1/ps.hat_1*prob_Y1_pred
  den <- (1-ps.hat_1)/ps.hat_1/ps.hat_1*prob_Y1_pred + (1-ps.hat_0)/ps.hat_0/ps.hat_0*(1-prob_Y1_pred)
  
  D <- rbinom(n1,1,0.5)
  D_opt <- (X1 + X2 + X3 < 0)*1
  D1 <- rep(0.5,n1)
  D2 <- D_opt * 0.4 + D1 * 0.6
  D3 <- D_opt * 0.7 + D1 * 0.3
  Value_eff_d1 <- mean(D1*(A1/ps.hat*Y1+(1-A1/ps.hat)*num/den))
  Value_eff_d2 <- mean(D2*(A1/ps.hat*Y1+(1-A1/ps.hat)*num/den))
  Value_eff_d3 <- mean(D3*(A1/ps.hat*Y1+(1-A1/ps.hat)*num/den))
  

  Value_nv1_d1 <- mean(D1*(A1/ps.hat_initial*Y1))
  Value_nv1_d2 <- mean(D2*(A1/ps.hat_initial*Y1))
  Value_nv1_d3 <- mean(D3*(A1/ps.hat_initial*Y1))
  
  Value_nv2_d1 <- mean(D1*(A1/ps.hat*Y1))
  Value_nv2_d2 <- mean(D2*(A1/ps.hat*Y1))
  Value_nv2_d3 <- mean(D3*(A1/ps.hat*Y1))
  
  fit_w <- gam(A1 ~X1+X2+X3,data = dat, family = "binomial")
  w.hat <- as.vector(predict(fit_w,dat,type = "response"))
  
  Value_nv3_d1 <- mean(D1*(A1/w.hat*(Y1-prob_Y1_pred)+prob_Y1_pred))
  Value_nv3_d2 <- mean(D2*(A1/w.hat*(Y1-prob_Y1_pred)+prob_Y1_pred))
  Value_nv3_d3 <- mean(D3*(A1/w.hat*(Y1-prob_Y1_pred)+prob_Y1_pred))
  
  
  c(Value_eff_d1,Value_eff_d2,Value_eff_d3,Value_nv1_d1,Value_nv1_d2,Value_nv1_d3,Value_nv2_d1,Value_nv2_d2,Value_nv2_d3,Value_nv3_d1,Value_nv3_d2,Value_nv3_d3,Value_true_d1,Value_true_d2,Value_true_d3)
}
stopImplicitCluster()
stopCluster(cl)

save(sim,file = "sim_binary_1000.RData")

