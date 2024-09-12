# url：https://zhuanlan.zhihu.com/p/689394559

library(tidyverse)
library(mlr3verse)
#使用mlr3内置的数据breast_cancer
task<-tsk("breast_cancer")#mlr3内置任务
data<-task$data()
breast_cancer<- as.data.frame(sapply(data[,2:10], function(x) as.numeric(as.character(x))))#把因子变量转化成数值变量，
breast_cancer$class<-data$class


library(mlr3verse)
library(mlr3)
task<-as_task_classif(breast_cancer,
                      target = "class")#重新构建任务

learners = list(
  learner_logreg = lrn("classif.log_reg", predict_type = "prob", predict_sets = c("train", "test")),
  learner_lda = lrn("classif.lda", predict_type = "prob",
                    predict_sets = c("train", "test")),
  learner_qda = lrn("classif.qda", predict_type = "prob",
                    predict_sets = c("train", "test")),
  learner_nb = lrn("classif.naive_bayes", predict_type = "prob",predict_sets = c("train", "test")),
  learner_knn = lrn("classif.kknn",predict_type = "prob"),
  learner_rpart = lrn("classif.rpart",predict_type = "prob"),
  learner_rf = lrn("classif.ranger", num.trees = 500,
                   predict_type = "prob"),
  learner_svm=lrn("classif.svm",predict_type="prob"),
  learner_xgb=lrn("classif.xgboost",predict_type="response"),
  learner_mul=lrn("classif.multinom",predict_type="response")
)

# benchmark()进行基准测试
rsmp_cv5 = rsmp("cv", folds = 5) #5折交叉验证
design = benchmark_grid(task, learners, rsmp_cv5)#创建一个超参数网格搜索的设计
design
future::plan("multisession") #并行计算任务
bmr = benchmark(design) #基准测试

# score()将返回每个学习器/任务/重采样组合的每个折叠的结果。
bmr$score()[, .(iteration, task_id, learner_id, classif.ce)]


data<-as.data.table(bmr$score(msr("classif.acc")))
library(ggpubr)
p<-ggplot(data,aes(x=learner_id,y=data$classif.acc,fill=learner_id))+
  geom_boxplot()+
  scale_fill_brewer(palette = "Set3")+
  theme_classic()+
  guides(fill="none")+
  theme(axis.text.x = element_text(size = 10,angle = 45,
                                   hjust = 1,vjust = 1))
#比较classif.log_reg,classif.multinom,classif.ranger,classif.svm是否有显著性差异
comparisons<-list(c("classif.log_reg","classif.multinom"),
                  c("classif.log_reg","classif.ranger"),
                  c("classif.log_reg","classif.svm"),
                  c("classif.multinom","classif.ranger"),
                  c("classif.multinom","classif.svm"),
                  c("classif.ranger","classif.svm"))

p+stat_compare_means(comparisons = comparisons,
                     method = "wilcox.test",
                     label="p.sighif")