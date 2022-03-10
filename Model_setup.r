#导包
library(e1071)
library(pROC)
library(ROCR)
library(ggplot2)
library(randomForest)

#导数据
dat = read.csv("D:/毕设/Wine_raw.csv")
dat = na.omit(dat)
names(dat)

#设定分类变量
dat$品牌<-as.factor(dat$品牌)
dat$商品产地<-as.factor(dat$商品产地)
dat$价格区间<-as.factor(dat$价格区间)
dat$新旧世界<-as.factor(dat$新旧世界)
dat$包装<-as.factor(dat$包装)

#划分数据集
set.seed(1)
sample = sample(nrow(dat), 2000)
dat_train = dat[sample, ]
dat_test = dat[-sample, ]

#径向核函数，SVM模型
t0<-proc.time()
tuned <-
  tune.svm(
    as.factor(价格区间) ~ .,
    data = dat_train,
    kernel = "radial",
    cost = 2^(1:9),
    gamma = c(1,2,3,4,5,6,7,8,9)
  )
summary(tuned)
plot(tuned)
svmfit.rad = tuned$best.model
summary(svmfit.rad)

table(svmfit.rad$fitted, dat_train$价格区间)
pred = predict(svmfit.rad, newdata = dat_test,decision.values = TRUE)
accuma=table(pred, dat_test$价格区间)
accu=(sum(diag(accuma)))/2416;accu
t1<-proc.time()
print(paste0("运算时间为: ",t1-t0,"秒"))

#保存模型, cost=64
save(svmfit.rad,file = "d:/毕设/svmfit.rad_v2.Rdata")

#随机森林方法
rf.redwine=randomForest(价格区间~.,data = dat_train,mtry=5,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
yhat.table=table(yhat.rf,dat_test[,"价格区间"]);yhat.table
sum(diag(yhat.table))/2416
