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

learner_logreg = lrn("classif.log_reg", predict_type = "prob",id="logreg")

learner_rf = lrn("classif.ranger", num.trees = 500,
                 predict_type = "prob",
                 mtry=to_tune(15,30),
                 min.node.size=to_tune(5,12),
                 max.depth=to_tune(3,9),
                 id="ranger")

learner_svm=lrn("classif.svm",predict_type="prob",
                type="C-classification",
                cost=to_tune(0.1,10),
                gamma=to_tune(0.1,10),
                kernel=to_tune(c("polynomial","radial","sigmoid")),
                degree=to_tune(1,3),
                id="svm")

learner_xgb=lrn("classif.xgboost",predict_type="response",
                eta=to_tune(0,1),
                gamma=to_tune(0,5),
                max_depth=to_tune(1,8),
                min_child_weight=to_tune(1,10),
                subsample=to_tune(0.5,1),
                colsample_bytree=to_tune(0.5,1),
                nrounds=to_tune(20,30),
                eval_metric=to_tune(c("merror","mlogloss")),
                id="xgboost")

learners = list(learner_logreg,learner_rf,learner_svm,learner_xgb)

granp<-ppl("branch",learners)
granp$plot()


#转化为图学习器
glearner<-as_learner(granp)
glearner


future::plan("multisession")
at<-auto_tuner(tnr("random_search"),
               learner = glearner,
               resampling = rsmp("cv",folds=3),
               measure = msr("classif.ce"),
               term_evals = 1000)

rr<-resample(task,at,rsmp("cv",folds=4),store_models = TRUE)
rr$aggregate()

extract_inner_tuning_results(rr)[,1:17]
