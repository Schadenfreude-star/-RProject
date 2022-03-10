#����
library(e1071)
library(pROC)
library(ROCR)
library(ggplot2)

#������
dat = read.csv("D:/����/���ģ�ͼ�����/wineDataML_3000AllNorm.csv")
dat = na.omit(dat)
names(dat)

#����ѵ���������Լ�
set.seed(1)
sample = sample(nrow(dat), 1000)
dat_train = dat[sample, ]
dat_test = dat[-sample, ]

#��svm��ʹ�����Ժ˺�������ͬ����(cost)�µ����Ų���
tuned <-
  tune.svm(
    as.factor(�۸�����) ~ .,
    data = dat_train,
    kernel = "linear",
    cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)
  )
summary(tuned)

#ʹ�����Ų�������ѵ��ģ��
svmfit = svm(
  �۸����� ~ .,
  data = dat_train,
  kernel = "linear",
  cost = 100,
  scale = FALSE,
  fitted=TRUE
)
#������
summary(svmfit)

#�б�Ƚϣ����岻��
table(svmfit$fitted, dat_train$�۸�����)
pred = predict(svmfit, newdata = dat_test,decision.values = TRUE)
table(pred, dat_test$�۸�����)

#����roc����
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
rocplot(fitted,dat_test[,"AHD"],main="Training Data",col="blue",auc=TRUE) #ISL�ϵķ��������Ƽ�
roc(
  dat_test[, "�۸�����"],
  fitted,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  print.auc = TRUE,
  print.auc.x = 45,
  col="red"
) #�����ҵķ������Ƽ�

#����ʹ�þ���˺���
tuned <-
  tune.svm(
    �۸����� ~ .,
    data = dat_train,
    kernel = "linear",
    cost = c(0.1, 1, 5, 10, 100),
    gamma = c(0.5,1,2,3,4)
  )
summary(tuned)

svmfit = svm(
  �۸����� ~ .,
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
  dat_test[, "�۸�����"],
  fitted,
  plot = TRUE,
  legacy.axes = TRUE,
  percent = TRUE,
  print.auc = TRUE,
  print.auc.x = 45,
  col="red"
) #�����ҵķ������Ƽ�

#PCA���ɷַ���
apply(dat_train, 2, mean)
apply(dat_train,2,var)

#�������ɷַ�������
pr.out=prcomp(dat_train,scale = TRUE)
#�鿴����������ݵ�����
names(pr.out)
#��׼��������ݾ�ֵ
pr.out$center
#��׼��������ݷ���
pr.out$scale
#���ɷ��غ���Ϣ
pr.out$rotation
#�鿴�����������Ϣ
dim(pr.out$x)
#˫��ͼ
biplot(pr.out,scale = 0)
#���ɷֱ�׼��
pr.out$sdev
#������ͳ̶�
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve

dat.pcaa=dat_train[,c("Ʒ��","��Ʒ����","�¾�����","Ʒ�ƽ���","��ׯ","���","�ڸ�")]

#PCA���ɷַ���
apply(dat.pcaa, 2, mean)
apply(dat.pcaa,2,var)

#�������ɷַ�������
pr.out=prcomp(dat.pcaa,scale = TRUE)
#�鿴����������ݵ�����
names(pr.out)
#��׼��������ݾ�ֵ
pr.out$center
#��׼��������ݷ���
pr.out$scale
#���ɷ��غ���Ϣ
pr.out$rotation
#�鿴�����������Ϣ
dim(pr.out$x)
#˫��ͼ
biplot(pr.out,scale = 0)
#���ɷֱ�׼��
pr.out$sdev
#������ͳ̶�
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve

#ϵͳ���෨
hc.complete=hclust(dist(dat_train),method = "complete")
hc.average=hclust(dist(dat_train),method = "average")
hc.single=hclust(dist(dat_train),method = "single")
plot(hc.complete,main = "Complete Linkage",xlab="",sub="",cex=.9,labels = FALSE)
plot(hc.average,main = "Complete Linkage",xlab="",sub="",cex=.9)
plot(hc.single,main = "Complete Linkage",xlab="",sub="",cex=.9)



library(tree)
dat_train$�۸�����=as.factor(dat_train$�۸�����)
dat_test$�۸�����=as.factor(dat_test$�۸�����)
tree.redwine=tree(�۸�����~.-�۸�����,dat_train)  #R��������Ҳ֧�֣�����Ҫ��˫����
summary(tree.redwine)
plot(tree.redwine)
text(tree.redwine,pretty = 0)
table(tree.redwine,dat_test[,"�۸�����"])

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
table(tree.pred,dat_test[,"�۸�����"])

library(randomForest)
rf.redwine=randomForest(�۸�����~.-�۸�����,data = dat_train,mtry=32,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
table(yhat.rf,dat_test[,"�۸�����"])
varImpPlot(rf.redwine)

rf.redwine=randomForest(�۸�����~.-�۸�����,data = dat_train,mtry=6,importance=TRUE)
rf.redwine
yhat.rf=predict(rf.redwine,newdata=dat_test)
table(yhat.rf,dat_test[,"�۸�����"])
varImpPlot(rf.redwine)
