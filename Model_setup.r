#����
library(e1071)
library(pROC)
library(ROCR)
library(ggplot2)
library(randomForest)

#������
dat = read.csv("D:/����/Wine_raw.csv")
dat = na.omit(dat)
names(dat)

#�趨�������
dat$Ʒ��<-as.factor(dat$Ʒ��)
dat$��Ʒ����<-as.factor(dat$��Ʒ����)
dat$�۸�����<-as.factor(dat$�۸�����)
dat$�¾�����<-as.factor(dat$�¾�����)
dat$��װ<-as.factor(dat$��װ)

#�������ݼ�
set.seed(1)
sample = sample(nrow(dat), 2000)
dat_train = dat[sample, ]
dat_test = dat[-sample, ]

#����˺�����SVMģ��
t0<-proc.time()
tuned <-
  tune.svm(
    as.factor(�۸�����) ~ .,
    data = dat_train,
    kernel = "radial",
    cost = 2^(1:9),
    gamma = c(1,2,3,4,5,6,7,8,9)
  )
summary(tuned)
plot(tuned)
svmfit.rad = tuned$best.model
summary(svmfit.rad)

table(svmfit.rad$fitted, dat_train$�۸�����)
pred = predict(svmfit.rad, newdata = dat_test,decision.values = TRUE)
accuma=table(pred, dat_test$�۸�����)
accu=(sum(diag(accuma)))/2416;accu
t1<-proc.time()
print(paste0("����ʱ��Ϊ: ",t1-t0,"��"))

#����ģ��, cost=64
save(svmfit.rad,file = "d:/����/svmfit.rad_v2.Rdata")

#���ɭ�ַ���
rf.redwine=randomForest(�۸�����~.,data = dat_train,mtry=5,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
yhat.table=table(yhat.rf,dat_test[,"�۸�����"]);yhat.table
sum(diag(yhat.table))/2416
