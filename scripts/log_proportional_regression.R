# https://zhuanlan.zhihu.com/p/687692129

library(tidyverse)
library(titanic)#使用内置的数据
data("titanic_train")#加载数据

titanicTib <- as_tibble(titanic_train)

fctrs <- c("Survived", "Sex", "Pclass")
titanicClean <- titanicTib %>%
  mutate_at(.vars = fctrs, .funs = factor) %>%#变量转化因子
  mutate(FamSize = SibSp + Parch) %>%
  mutate(EmbarkedFactor=factor(.$Embarked,levels = c("C","Q","S"))) %>%
  select(Survived, Pclass, Sex, Fare, FamSize)
titanicClean

library(tidyverse)
titanicUntidy <- gather(titanicClean, key = "Variable", value = "Value",-Survived)#把宽数据转化长数据，有利于图形展示
titanicUntidy

titanicUntidy %>%
  filter(Variable != "Pclass" & Variable != "Sex") %>%
  ggplot(aes(Survived, as.numeric(Value))) +#把字符型数据转化为数值型数据
  facet_wrap(~ Variable, scales = "free_y") +#分面
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +#小提琴图
  theme_bw()

titanicUntidy %>%
  filter(Variable == "Pclass" | Variable == "Sex") %>%
  ggplot(aes(Value, fill = Survived)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_bar(position = "fill") +#条形图
  theme_bw()

library(mlr3verse)
library(mlr3fselect)##用于特征选择

set.seed(12345)
task<-as_task_classif(titanicClean,
                      target = "Survived")#任务

learner<-lrn("classif.log_reg",#学习器
             predict_type="prob")#设置模型预测类型为概率预测，即输出每个类别的概率


afs= auto_fselector(fselector=fs("exhaustive_search"),#穷举搜索
                    learner = learner,#学习器
                    resampling = rsmp("cv", folds = 4),#内部交叉验证使用4折交叉验证。
                    measure = msr("classif.auc"), #评估特征选择性能的指标AUC
                    term_evals = 20,
                    store_models = TRUE)

rr_afs<-resample(task,afs,rsmp("cv",folds=5),store_models = TRUE)
extract_inner_fselect_results(rr_afs)

extract_inner_fselect_archives(rr_afs)[,1:6]

rr_afs$aggregate(msr("classif.auc"))

rr_afs$score(msr("classif.auc"))[,.(task_id,resampling_id,iteration,classif.auc)]

# 对特征选择结果进行评估
grid = benchmark_grid(task, list(afs,learner ), rsmp("cv", folds = 3))
bmr = benchmark(grid)$aggregate(msr("classif.auc"))
as.data.table(bmr)[, .(learner_id, classif.auc)]

# 使用嵌套特征选择fselect_nested()函数，实现特征选择
rr = fselect_nested(fselector = fs("exhaustive_search"),#穷举搜索
                    task = task,#任务
                    learner = learner,#学习器
                    inner_resampling = rsmp("cv", folds = 4),#内部交叉验证使用4折交叉验证。
                    outer_resampling = rsmp("cv", folds = 5),#外部交叉验证使用5折交叉验证。
                    measure = msr("classif.auc"), #评估特征选择性能的指标AUC
                    term_evals = 20,
                    store_models = TRUE)
extract_inner_fselect_results(rr)

task$select(c("Fare","Pclass","Sex"))
learner$train(task)
predictions<-learner$predict(task)
predictions$confusion#查看混淆矩阵

predictions

predictions$score(msr("classif.acc"))##模型的准确率
learner$model#模型

dat<-data.frame(Pclass=c("1","2","3","1","2","3"),
                Sex=c("female","female","female","male","male","male"),
                Fare=c(29,150,79,19,8,59))#新建数据集
learner$predict_newdata(dat)
