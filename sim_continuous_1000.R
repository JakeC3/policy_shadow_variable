library(nleqslv)
library(MASS)
library(randomForest)
library(mgcv)
set.seed(2023)
sigma <- matrix(c(1,-0.25,-0.25,-0.25,1,-0.25,-0.25,-0.25,1),3,3)
n2 <- 10^6
c2 <-  mvrnorm(n2,c(1,-1,0),sigma)
x1 <- c2[,1]
x2 <- c2[,2]
x3 <- c2[,3] #shadow
y1 <-  4*(2*x1 - x1^2 -x2 + x3^2)
d_opt <- (2*x1 - x1^2 -x2 + x3^2 >0)*1
d1 <- rep(0.5,n2)
d2 <- d_opt * 0.3 + d1 * 0.7
d3 <- d_opt * 0.6 + d1 * 0.4
Value_true_d1 <- mean(d1*y1)
Value_true_d2 <- mean(d2*y1)
Value_true_d3 <- mean(d3*y1)

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
  set.seed(k+2000)
  n1 <- 1000
  C1 <- matrix(0,n1,3)
  colnames(C1) <- c("X1","X2","X3")
  C1 <- mvrnorm(n1,c(1,-1,0),sigma)
  X1 <- C1[,1] 
  X2 <- C1[,2]
  X3 <- C1[,3] #shadow variable
  Y1 <- 4*(2*X1 - X1^2 -X2 + X3^2) + rnorm(n1,0,0.5)
  ps1 <- 1/(1+exp(-(-0.5+X1+X2+0.1*Y1)))
  A1 <- sapply(ps1,function(x) rbinom(1,1,x))
  
  CM1 <- cbind(rep(1,n1),X1,X2,Y1)
  M1 <- cbind(rep(1,n1),C1)
  eta.hat_initial <- nleqslv(x=c(-0.5,1,1,.1),fn=etaFun,cov_matrix=CM1,moments=M1,trt=A1)$x
  ps.hat_initial <- as.vector(1 / (1+exp(-CM1%*%eta.hat_initial)))
  
  index_trt1 <- which(A1==1)
  pseudo1 <- ((1-ps.hat_initial) / ps.hat_initial)[index_trt1]
  pseudo2 <- ((1-ps.hat_initial) / ps.hat_initial * X1)[index_trt1]
  pseudo3 <- ((1-ps.hat_initial) / ps.hat_initial * X2)[index_trt1]
  pseudo4 <- ((1-ps.hat_initial) / ps.hat_initial * Y1)[index_trt1]
  pseudo5 <- ((ps.hat_initial-1) / ps.hat_initial / ps.hat_initial)[index_trt1]

  C1_trt1 <- C1[index_trt1,]
  data_trt1 <- data.frame(pseudo1, C1_trt1)
  model1 <- randomForest(pseudo1~.,data=data_trt1,ntree=1000)
  data_trt1 <- data.frame(pseudo2, C1_trt1)
  model2 <- randomForest(pseudo2~.,data=data_trt1,ntree=1000)
  data_trt1 <- data.frame(pseudo3, C1_trt1)
  model3 <- randomForest(pseudo3~.,data=data_trt1,ntree=1000)
  data_trt1 <- data.frame(pseudo4, C1_trt1)
  model4 <- randomForest(pseudo4~.,data=data_trt1,ntree=1000)
  data_trt1 <- data.frame(pseudo5, C1_trt1)
  model5 <- randomForest(pseudo5~.,data=data_trt1,ntree=1000)
  
  pseudo1.hat <- rep(0,n1)
  pseudo2.hat <- rep(0,n1)
  pseudo3.hat <- rep(0,n1)
  pseudo4.hat <- rep(0,n1)
  pseudo5.hat <- rep(0,n1)
  
  data_trt0 <- data.frame(C1[-index_trt1,])
  pseudo1.hat[index_trt1] <- model1$predicted
  pseudo2.hat[index_trt1] <- model2$predicted
  pseudo3.hat[index_trt1] <- model3$predicted
  pseudo4.hat[index_trt1] <- model4$predicted
  pseudo5.hat[index_trt1] <- model5$predicted
  
  pseudo1.hat[-index_trt1] <- predict(model1,newdata=data_trt0)
  pseudo2.hat[-index_trt1] <- predict(model2,newdata=data_trt0)
  pseudo3.hat[-index_trt1] <- predict(model3,newdata=data_trt0)
  pseudo4.hat[-index_trt1] <- predict(model4,newdata=data_trt0)
  pseudo5.hat[-index_trt1] <- predict(model5,newdata=data_trt0)
  
  M_pseudo = cbind(pseudo1.hat,pseudo2.hat,pseudo3.hat,pseudo4.hat) / pseudo5.hat
  eta.hat <- nleqslv(x=c(-0.5,1,1,.1),fn=etaFun,cov_matrix=CM1,moments=M_pseudo,trt=A1)$x
  ps.hat <- as.vector(1 / (1+exp(-CM1%*%eta.hat)))
  
  pseudo6.hat <- rep(0,n1)
  pseudo7.hat <- rep(0,n1)
  
  pseudo6 <- (Y1 * (1-ps.hat) / ps.hat /ps.hat)[index_trt1]
  data_trt1 <- data.frame(pseudo6, C1_trt1)
  model6 <- randomForest(pseudo6~.,data=data_trt1,ntree=1000)
  pseudo6.hat[index_trt1] <- model6$predicted
  pseudo6.hat[-index_trt1] <- predict(model6,newdata=data_trt0)
  
  pseudo7 <- ((1-ps.hat) / ps.hat / ps.hat)[index_trt1]
  data_trt1 <- data.frame(pseudo7, C1_trt1)
  model7 <- randomForest(pseudo7~.,data=data_trt1,ntree=1000)
  pseudo7.hat[index_trt1] <- model7$predicted
  pseudo7.hat[-index_trt1] <- predict(model7,newdata=data_trt0)
  
  D <- rbinom(n1,1,0.5)
  D_opt <- (2*X1 - X1^2 -X2 + X3^2 >0)*1
  D1 <- rep(0.5,n1)
  D2 <- D_opt * 0.3 + D1 * 0.7
  D3 <- D_opt * 0.6 + D1 * 0.4
  
  Value_eff_d1 <- mean(D1*(A1/ps.hat*Y1+(1-A1/ps.hat)*pseudo6.hat/pseudo7.hat))
  Value_eff_d2 <- mean(D2*(A1/ps.hat*Y1+(1-A1/ps.hat)*pseudo6.hat/pseudo7.hat))
  Value_eff_d3 <- mean(D3*(A1/ps.hat*Y1+(1-A1/ps.hat)*pseudo6.hat/pseudo7.hat))
  
  Value_nv1_d1 <- mean(D1*(A1/ps.hat_initial*Y1))
  Value_nv1_d2 <- mean(D2*(A1/ps.hat_initial*Y1))
  Value_nv1_d3 <- mean(D3*(A1/ps.hat_initial*Y1))
  
  Value_nv2_d1 <- mean(D1*(A1/ps.hat*Y1))
  Value_nv2_d2 <- mean(D2*(A1/ps.hat*Y1))
  Value_nv2_d3 <- mean(D3*(A1/ps.hat*Y1))
  
  Y1_trt1 <- Y1[index_trt1]
  data_trt1 <- data.frame(Y1_trt1, C1_trt1)
  model8 <- randomForest(Y1_trt1~.,data=data_trt1,ntree=1000)
  Y1.hat <- rep(0,n1)
  Y1.hat[index_trt1] <- model8$predicted
  Y1.hat[-index_trt1] <- predict(model8,newdata=data_trt0)
  
  dat <- data.frame(A1,C1)
  model9 <- gam(A1 ~X1+X2+X3,data = dat, family = "binomial")
  w.hat <- as.vector(predict(model9,dat,type = "response"))
  
  Value_nv3_d1 <- mean(D1*(A1/w.hat*(Y1-Y1.hat)+Y1.hat))
  Value_nv3_d2 <- mean(D2*(A1/w.hat*(Y1-Y1.hat)+Y1.hat))
  Value_nv3_d3 <- mean(D3*(A1/w.hat*(Y1-Y1.hat)+Y1.hat))
  
  
  
  c(Value_eff_d1,Value_eff_d2,Value_eff_d3,Value_nv1_d1,Value_nv1_d2,Value_nv1_d3,Value_nv2_d1,Value_nv2_d2,Value_nv2_d3,Value_nv3_d1,Value_nv3_d2,Value_nv3_d3,Value_true_d1,Value_true_d2,Value_true_d3)
}
stopImplicitCluster()
stopCluster(cl)

save(sim,file = "sim_continuous_1000.RData")
