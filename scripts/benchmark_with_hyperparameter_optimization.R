# url:链接：https://zhuanlan.zhihu.com/p/689397085

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
split<-partition(task)
task$set_row_roles(split$test,"use")#留三分之一做测试集


#在学习器前面加一个特征选择，po("filter")自动化基于过滤器的特征选择
po_filter = po("filter", 
               filter = flt("information_gain"),#过滤器方法选择信息增益
               filter.nfeat = to_tune(1, 9))#选择的特征数量作为超参数进行调整


# 改成循环跑


# 4.1 逻辑回归模型
logreg = lrn("classif.log_reg", predict_type = "prob")#学习器
logreg_fliter<-as_learner(po_filter %>>% logreg)#特征选择+学习器

#tune()对学习器做超参数调整
instance_logreg<-tune(tuner = tnr("random_search"),
                      task=task,
                      learner = logreg_fliter,
                      resampling = rsmp("cv",folds=3),
                      measures = msr("classif.ce"),
                      terminator =trm("evals",n_evals=20,k=0)
)
instance_logreg$result

#最优超参数更新学习器
logreg_fliter$param_set$values<-instance_logreg$result_learner_param_vals
#训练集训练模型
logreg_fliter$train(task)
#测试集验证模型
predictions_logreg<-logreg_fliter$predict(task,row_ids = split$test)
k1<-predictions_logreg$score(c(msr("classif.ce"),
                               msr("classif.acc"),
                               msr("classif.auc")))
k1
