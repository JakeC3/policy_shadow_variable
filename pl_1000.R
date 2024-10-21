library(nleqslv)
library(MASS)
library(randomForest)
library(mgcv)
library(policytree)
library(DiagrammeR)
set.seed(2023)
sigma <- matrix(c(1,-0.25,-0.25,-0.25,1,-0.25,-0.25,-0.25,1),3,3)
n2 <- 10^5
c2 <-  mvrnorm(n2,c(1,-1,0),sigma)
x1 <- c2[,1]
x2 <- c2[,2]
x3 <- c2[,3] #shadow
# y1 <-  4*(2*x1 - x1^2 -x2 + x3^2)
y1 <-  4*(2*x1 - 1.5*x1^2 -x2 + 0.5*x3^2)
y0 <- 0
scores <- data.frame(y0, y1)
colnames(scores) <- c("trt_0","trt_1")
x <- cbind(x1,x2,x3)
tree_opt <- policy_tree(x, scores, depth =2,split.step = 10)
d_opt <- predict(tree_opt,x)
value_opt <- mean((d_opt-1)*y1)

etaFun <- function(eta, cov_matrix, moments, trt){ 
  ps <- as.vector(1 / (1+exp(-cov_matrix%*%eta)))
  colSums((1-trt/ps)*moments)
}

library(foreach)
library(doParallel)
no_cores <- detectCores(logical=F)
cl <- makeCluster(no_cores)
registerDoParallel(cl)

sim <- foreach(k=1:300,.combine='cbind',.packages = c("MASS","randomForest","nleqslv","mgcv","policytree"))%dopar% {
  set.seed(k+1500)
  n1 <- 1000
  C1 <- matrix(0,n1,3)
  colnames(C1) <- c("X1","X2","X3")
  C1 <- mvrnorm(n1,c(1,-1,0),sigma)
  X1 <- C1[,1] 
  X2 <- C1[,2]
  X3 <- C1[,3] #shadow variable
  Y1 <- 4*(2*X1 - 1.5*X1^2 -X2 + 0.5*X3^2) + rnorm(n1,0,0.25)
  # Y1 <- 4*(2*X1 - X1^2 -X2 + X3^2) + rnorm(n1,0,0.5)
  ps1 <- 1/(1+exp(-(-0.5+X1+X2+0.15*Y1)))
  A1 <- sapply(ps1,function(x) rbinom(1,1,x))
  
  CM1 <- cbind(rep(1,n1),X1,X2,Y1)
  M1 <- cbind(rep(1,n1),C1)
  eta.hat_initial <- nleqslv(x=c(-0.5,1,1,.15),fn=etaFun,cov_matrix=CM1,moments=M1,trt=A1)$x
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
  eta.hat <- nleqslv(x=c(-0.5,1,1,.15),fn=etaFun,cov_matrix=CM1,moments=M_pseudo,trt=A1)$x
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
  
  scores_eff <- data.frame(rep(0,n1),A1/ps.hat*Y1+(1-A1/ps.hat)*pseudo6.hat/pseudo7.hat)
  colnames(scores_eff) <- c("trt_0","trt_1")
  tree_opt_eff <- policy_tree(C1, scores_eff, depth =2)
  d_opt_eff <- predict(tree_opt_eff,x)
  pcd_eff <- sum(d_opt==d_opt_eff) / n2
  value_eff <- mean((d_opt_eff-1)*y1)

  scores_nv1 <- data.frame(rep(0,n1),A1/ps.hat_initial*Y1)
  colnames(scores_nv1) <- c("trt_0","trt_1")
  tree_opt_nv1 <- policy_tree(C1, scores_nv1, depth =2)
  d_opt_nv1 <- predict(tree_opt_nv1,x)
  pcd_nv1 <- sum(d_opt==d_opt_nv1) / n2
  value_nv1 <- mean((d_opt_nv1-1)*y1)
  
  scores_nv2 <- data.frame(rep(0,n1),A1/ps.hat*Y1)
  colnames(scores_nv2) <- c("trt_0","trt_1")
  tree_opt_nv2 <- policy_tree(C1, scores_nv2, depth =2)
  d_opt_nv2 <- predict(tree_opt_nv2,x)
  pcd_nv2 <- sum(d_opt==d_opt_nv2) / n2
  value_nv2 <- mean((d_opt_nv2-1)*y1)
  
  Y1_trt1 <- Y1[index_trt1]
  data_trt1 <- data.frame(Y1_trt1, C1_trt1)
  model8 <- randomForest(Y1_trt1~.,data=data_trt1,ntree=1000)
  Y1.hat <- rep(0,n1)
  Y1.hat[index_trt1] <- model8$predicted
  Y1.hat[-index_trt1] <- predict(model8,newdata=data_trt0)
  
  dat <- data.frame(A1,C1)
  model9 <- gam(A1 ~X1+X2+X3,data = dat, family = "binomial")
  w.hat <- as.vector(predict(model9,dat,type = "response"))
  
  scores_nv3 <- data.frame(rep(0,n1),A1/w.hat*(Y1-Y1.hat)+Y1.hat)
  colnames(scores_nv3) <- c("trt_0","trt_1")
  tree_opt_nv3 <- policy_tree(C1, scores_nv3, depth = 2)
  d_opt_nv3 <- predict(tree_opt_nv3,x)
  pcd_nv3 <- sum(d_opt==d_opt_nv3) / n2
  value_nv3 <- mean((d_opt_nv3-1)*y1)
  
  
  c(pcd_eff,pcd_nv1,pcd_nv2,pcd_nv3,value_eff,value_nv1,value_nv2,value_nv3,value_opt)
}
stopImplicitCluster()
stopCluster(cl)

save(sim,file = "pl_1000.RData")
