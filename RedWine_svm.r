#导包
library(e1071)
library(pROC)
library(ROCR)
library(ggplot2)

#导数据
dat = read.csv("D:/毕设/红酒模型及数据/wineDataML_3000AllNorm.csv")
dat = na.omit(dat)
names(dat)

#划分训练集、测试集
set.seed(1)
sample = sample(nrow(dat), 1000)
dat_train = dat[sample, ]
dat_test = dat[-sample, ]

#找svm，使用线性核函数、不同参数(cost)下的最优参数
tuned <-
  tune.svm(
    as.factor(价格区间) ~ .,
    data = dat_train,
    kernel = "linear",
    cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)
  )
summary(tuned)

#使用最优参数进行训练模型
svmfit = svm(
  价格区间 ~ .,
  data = dat_train,
  kernel = "linear",
  cost = 100,
  scale = FALSE,
  fitted=TRUE
)
#输出结果
summary(svmfit)

#列表比较（意义不大）
table(svmfit$fitted, dat_train$价格区间)
pred = predict(svmfit, newdata = dat_test,decision.values = TRUE)
table(pred, dat_test$价格区间)

#绘制roc曲线
fitted=attributes(predict(svmfit,dat_test,decision.values  =TRUE))$decision.values
summary(fitted)

rocplot=function(pred,truth,...){
  predob=prediction(pred,truth)
  perf=performance(predob,"tpr","fpr")
  auc=performance(predob,"auc")
  auc=unlist(slot(auc,"y.values"))
  plot(perf,print.auc=TRUE,...)
  print(auc)
}
par(mfrow=c(1,2))
rocplot(fitted,dat_test[,"AHD"],main="Training Data",col="blue",auc=TRUE) #ISL上的方法，不推荐
roc(
  dat_test[, "价格区间"],
  fitted,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  print.auc = TRUE,
  print.auc.x = 45,
  col="red"
) #网上找的方法，推荐

#决定使用径向核函数
tuned <-
  tune.svm(
    价格区间 ~ .,
    data = dat_train,
    kernel = "linear",
    cost = c(0.1, 1, 5, 10, 100),
    gamma = c(0.5,1,2,3,4)
  )
summary(tuned)

svmfit = svm(
  价格区间 ~ .,
  data = dat_train,
  kernel = "radial",
  cost = 100,
  gamma = 0.03125,
  scale = FALSE,
  fitted=TRUE
)
summary(svmfit)

fitted=attributes(predict(svmfit,dat_test,decision.values  =TRUE))$decision.values
summary(fitted)
roc(
  dat_test[, "价格区间"],
  fitted,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  print.auc = TRUE,
  print.auc.x = 45,
  col="red"
) #网上找的方法，推荐

#PCA主成分分析
apply(dat_train, 2, mean)
apply(dat_train,2,var)

#代入主成分分析函数
pr.out=prcomp(dat_train,scale = TRUE)
#查看函数输出内容的类型
names(pr.out)
#标准化后的数据均值
pr.out$center
#标准化后的数据方差
pr.out$scale
#主成分载荷信息
pr.out$rotation
#查看特征矩阵的信息
dim(pr.out$x)
#双标图
biplot(pr.out,scale = 0)
#主成分标准差
pr.out$sdev
#输出解释程度
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve

dat.pcaa=dat_train[,c("品牌","商品产地","新旧世界","品酒进阶","名庄","甜度","口感")]

#PCA主成分分析
apply(dat.pcaa, 2, mean)
apply(dat.pcaa,2,var)

#代入主成分分析函数
pr.out=prcomp(dat.pcaa,scale = TRUE)
#查看函数输出内容的类型
names(pr.out)
#标准化后的数据均值
pr.out$center
#标准化后的数据方差
pr.out$scale
#主成分载荷信息
pr.out$rotation
#查看特征矩阵的信息
dim(pr.out$x)
#双标图
biplot(pr.out,scale = 0)
#主成分标准差
pr.out$sdev
#输出解释程度
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve

#系统聚类法
hc.complete=hclust(dist(dat_train),method = "complete")
hc.average=hclust(dist(dat_train),method = "average")
hc.single=hclust(dist(dat_train),method = "single")
plot(hc.complete,main = "Complete Linkage",xlab="",sub="",cex=.9,labels = FALSE)
plot(hc.average,main = "Complete Linkage",xlab="",sub="",cex=.9)
plot(hc.single,main = "Complete Linkage",xlab="",sub="",cex=.9)



library(tree)
dat_train$价格区间=as.factor(dat_train$价格区间)
dat_test$价格区间=as.factor(dat_test$价格区间)
tree.redwine=tree(价格区间~.-价格区间,dat_train)  #R语言中文也支持，不需要加双引号
summary(tree.redwine)
plot(tree.redwine)
text(tree.redwine,pretty = 0)
table(tree.redwine,dat_test[,"价格区间"])

cv.redwine=cv.tree(tree.redwine,FUN=prune.misclass)
names(cv.redwine)
cv.redwine

par(mfrow=c(1,2))
plot(cv.redwine$size,cv.redwine$dev,type='b')
plot(cv.redwine$k,cv.redwine$dev,type='b')

prune.redwine=prune.misclass(tree.redwine,best=6)
plot(prune.redwine)
text(prune.redwine,pretty=0)
tree.pred=predict(prune.redwine,dat_test,type="class")
table(tree.pred,dat_test[,"价格区间"])

library(randomForest)
rf.redwine=randomForest(价格区间~.-价格区间,data = dat_train,mtry=32,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
table(yhat.rf,dat_test[,"价格区间"])
varImpPlot(rf.redwine)

rf.redwine=randomForest(价格区间~.-价格区间,data = dat_train,mtry=6,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
table(yhat.rf,dat_test[,"价格区间"])
varImpPlot(rf.redwine)
