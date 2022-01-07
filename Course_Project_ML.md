---
title: 'Prediction Assignment Writeup'
subtitle: 'Author: S. Jiménez Ariza'
output:
  html_document:
    keep_md: yes
  pdf_document: default
---
## Executive Summary
This document presents a model to predict how people exercise. To this aim, an analysis of the data from accelerometers on the belt, forearm, arm, and dumbbell of 6 people is performed. As described in Velloso et al. (2013), the five considered classes are: A, exactly according to the specification; B, throwing the elbows to the front; C, lifting the dumbbell only halfway; D, lowering the dumbbell only halfway; and E, throwing the hips to the front.

Two models are tested following two algorithms (i.e., random forest and boosting). In each case, a cross-validation method is included in the development of the model. Additionally, the training data is divided into two sets to test the results. Due to the better performance of random forest, this model is selected to predict the class that describes the exercise. 

## Loading the Data and preprocessing

The data is loaded and pre-processed to eliminate columns that will not contribute to the prediction. This includes: (1) columns with data to characterize the user, (2) columns with a majority of absent values (using a threshold of at list 95% of the values being different to NAs), and (3) columns with zero or near to zero variance.


```r
library(knitr);library(magrittr);library(nnet);library(dplyr);library(caret)
pml_train<-read.csv(file="pml-training.csv",header=TRUE)
pml_test<-read.csv(file="pml-testing.csv",header=TRUE) # Load the data
kable(summary(as.factor(pml_train$classe)))## table to see the data
```



|   |    x|
|:--|----:|
|A  | 5580|
|B  | 3797|
|C  | 3422|
|D  | 3216|
|E  | 3607|


```r
#Columns without data to analyse movement are discarded

pml_train_s<-pml_train[,8:dim(pml_train)[2]]
pml_test_s<-pml_test[,8:dim(pml_test)[2]]

#Columns without enough information are discarded
pml_train_s<-pml_train_s %>% mutate_at(c(1:dim(pml_train_s)[2]-1), as.numeric)
pml_test_s<-pml_test_s %>% mutate_at(c(1:dim(pml_test_s)[2]), as.numeric)
col_data<-colSums(is.na(pml_train_s)) <= dim(pml_train)[1]*0.95
pml_train_s<-pml_train_s[,col_data]
pml_test_s<-pml_test_s[,col_data]

#Variables with variance zero or near to zero are discarded
var_0<-nearZeroVar(pml_train_s, saveMetrics = TRUE)
pml_train_s<-pml_train_s[,!(var_0$nzv |var_0$zeroVar)]
pml_test_s<-pml_test_s[,!(var_0$nzv |var_0$zeroVar)]
```

## Model
To develop the model, two subsets are created to perform cross-validation, which will provide additional information to estimate the out of sample error. Following this, two algorithms are used. In each case, a cross-validation method is specified with the trainControl() function along with the number of cross-validation folds (i.e., 3). The end of the section presents the confusion matrices and accuracy for the developed models and the tested predictions.

1. Data Partition


```r
set.seed(12345)
training<-createDataPartition(y=pml_train_s$classe, p=0.7,list=FALSE)
pml_train_train<-pml_train_s[training,]
pml_train_test<-pml_train_s[-training,]
```

2. Model building

2.1. Random forest


```r
# Random forest
set.seed(12345)
mod1<-train(classe~.,method="rf",data=pml_train_train,trControl=trainControl(method="cv",number=3))
rf_cm<-confusionMatrix(mod1)
```

2.2. Boosting


```r
#Boosting
set.seed(12345)
mod2<-train(classe ~ ., method="gbm",data=pml_train_train,verbose=FALSE,trControl=trainControl(method="cv",number=3))
b_cm<-confusionMatrix(mod2)
```

3.3. Summary models

```r
#Boosting
results<- data.frame(
        Model = c('Random Forest', 'Boosting'),
        Accuracy = rbind(mod1$results[which(as.numeric(mod1$bestTune) == mod1$results),2], max(mod2$results[5]))
)
```

The confusion matrix of the model developed with the random forest algorithm is:



|   |          A|          B|          C|          D|          E|
|:--|----------:|----------:|----------:|----------:|----------:|
|A  | 28.3759191|  0.2256679|  0.0000000|  0.0000000|  0.0072796|
|B  |  0.0291184| 19.0725777|  0.1601514|  0.0145592|  0.0145592|
|C  |  0.0072796|  0.0509573| 17.2089976|  0.2256679|  0.0509573|
|D  |  0.0145592|  0.0000000|  0.0727961| 16.1316153|  0.0727961|
|E  |  0.0072796|  0.0000000|  0.0000000|  0.0218388| 18.2354226|



And the confusion matrix developed with the boosting algorithm is:



|   |          A|          B|          C|          D|          E|
|:--|----------:|----------:|----------:|----------:|----------:|
|A  | 27.9173036|  0.6478853|  0.0072796|  0.0072796|  0.0291184|
|B  |  0.2911844| 18.1189488|  0.5750892|  0.0800757|  0.1965495|
|C  |  0.1237534|  0.5168523| 16.6266288|  0.5314115|  0.1601514|
|D  |  0.0873553|  0.0436777|  0.1892699| 15.6438815|  0.2693456|
|E  |  0.0145592|  0.0218388|  0.0436777|  0.1310330| 17.7258499|



As a result the accuracy of both models is presented in the next table:



|Model         |  Accuracy|
|:-------------|---------:|
|Random Forest | 0.9902452|
|Boosting      | 0.9603261|



4. Testing


```r
# Random forest
pred1<-predict(mod1,pml_train_test)
rf_cm_test <- confusionMatrix(pred1, as.factor(pml_train_test$classe))
#sqrt(sum((pred1!=pml_train_test$classe)^2))
#Boosting
pred2<-predict(mod2,pml_train_test)
b_cm_test <- confusionMatrix(pred2, as.factor(pml_train_test$classe))
#sqrt(sum((pred2!=pml_train_test$classe)^2))
results_test<- data.frame(
        Model = c('Random Forest', 'Boosting'),
        Accuracy = rbind(rf_cm_test$overall[1], b_cm_test$overall[1]),
        AccuracyLower = rbind(rf_cm_test$overall[3], b_cm_test$overall[3]),
        AccuracyUpper = rbind(rf_cm_test$overall[4], b_cm_test$overall[4])
)
```

When the model is tested with the available data the resulting confusion matrices are:

1. Random forest


|   |    A|    B|    C|   D|    E|
|:--|----:|----:|----:|---:|----:|
|A  | 1672|    7|    0|   0|    0|
|B  |    1| 1129|    4|   0|    0|
|C  |    1|    3| 1019|   7|    1|
|D  |    0|    0|    3| 956|    1|
|E  |    0|    0|    0|   1| 1080|


2. Boosting


|   |    A|    B|   C|   D|    E|
|:--|----:|----:|---:|---:|----:|
|A  | 1640|   45|   0|   2|    1|
|B  |   27| 1058|  36|   2|   12|
|C  |    4|   32| 978|  39|    6|
|D  |    3|    0|  11| 916|    9|
|E  |    0|    4|   1|   5| 1054|



The accuracy of both models is summarized in the next table:


|Model         |  Accuracy| AccuracyLower| AccuracyUpper|
|:-------------|---------:|-------------:|-------------:|
|Random Forest | 0.9950722|     0.9929305|     0.9966974|
|Boosting      | 0.9593883|     0.9540274|     0.9642874|



Given the results the model developed with the random forest algorithm is selected and the expected out of sample error is 0.493%

## Prediction


```r
# Random forest
test1<-predict(mod1,pml_test)
prediction1<-data.frame(pml_test$problem_id,test1)
```

Given the performance of the models the one developed with random forest is selected. The predicted values are presented in the next table:



| pml_test.problem_id|test1 |
|-------------------:|:-----|
|                   1|B     |
|                   2|A     |
|                   3|B     |
|                   4|A     |
|                   5|A     |
|                   6|E     |
|                   7|D     |
|                   8|B     |
|                   9|A     |
|                  10|A     |
|                  11|B     |
|                  12|C     |
|                  13|B     |
|                  14|A     |
|                  15|E     |
|                  16|E     |
|                  17|A     |
|                  18|B     |
|                  19|B     |
|                  20|B     |



## Conclusions

Two models were developed with two different algorithms. Both performed well, having high accuracy. The model developed with random forest was selected given a better performance and 20 values were estimated according to it. The out of sample error estimated with cross-validation was  0.493% . Similarly, the estimated accuracy was  99.507%

## References

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

## Appendix

The characteristics of the model are summarized in this section:

1. Results:


| mtry|  Accuracy|     Kappa| AccuracySD|   KappaSD|
|----:|---------:|---------:|----------:|---------:|
|    2| 0.9892990| 0.9864608|  0.0020831| 0.0026369|
|   27| 0.9902452| 0.9876590|  0.0015353| 0.0019440|
|   52| 0.9827471| 0.9781754|  0.0015786| 0.0019999|


2. Number of variables tried at each split (final model): 27
3. Importance of the variables (final model):




|                     |          x|
|:--------------------|----------:|
|roll_belt            | 1402.76015|
|pitch_forearm        |  845.30567|
|yaw_belt             |  792.87366|
|magnet_dumbbell_y    |  658.72310|
|pitch_belt           |  636.31494|
|roll_forearm         |  622.41238|
|magnet_dumbbell_z    |  617.94480|
|accel_dumbbell_y     |  331.20540|
|accel_forearm_x      |  277.22321|
|magnet_dumbbell_x    |  247.63889|
|roll_dumbbell        |  242.02210|
|magnet_belt_z        |  241.04866|
|accel_belt_z         |  224.33601|
|magnet_forearm_z     |  223.09301|
|accel_dumbbell_z     |  210.12226|
|total_accel_dumbbell |  192.30925|
|magnet_belt_y        |  190.93196|
|gyros_belt_z         |  175.86566|
|yaw_arm              |  170.52777|
|magnet_belt_x        |  156.92544|
|yaw_dumbbell         |  138.07209|
|roll_arm             |  134.77763|
|gyros_dumbbell_y     |  131.12558|
|accel_forearm_z      |  131.07452|
|magnet_forearm_y     |  119.60265|
|accel_dumbbell_x     |  107.54590|
|accel_arm_x          |  106.91745|
|magnet_arm_y         |  105.51393|
|magnet_arm_x         |  101.36506|
|yaw_forearm          |   96.38686|
|magnet_forearm_x     |   96.35138|
|magnet_arm_z         |   94.42664|
|pitch_arm            |   88.74816|
|gyros_arm_y          |   81.06740|
|accel_forearm_y      |   72.88501|
|pitch_dumbbell       |   71.65953|
|accel_arm_y          |   69.61399|
|gyros_dumbbell_x     |   64.61037|
|gyros_arm_x          |   63.68345|
|total_accel_belt     |   54.26396|
|gyros_forearm_y      |   53.82033|
|accel_arm_z          |   52.44174|
|total_accel_arm      |   48.06040|
|gyros_belt_y         |   43.51110|
|total_accel_forearm  |   40.96057|
|gyros_dumbbell_z     |   39.83620|
|gyros_belt_x         |   38.08953|
|accel_belt_x         |   36.41121|
|gyros_forearm_z      |   34.59545|
|accel_belt_y         |   33.44368|
|gyros_forearm_x      |   25.12909|
|gyros_arm_z          |   24.50174|



4. Confusion Matrix (final model):



|   |    A|    B|    C|    D|    E| class.error|
|:--|----:|----:|----:|----:|----:|-----------:|
|A  | 3899|    5|    0|    0|    2|   0.0017921|
|B  |   19| 2630|    9|    0|    0|   0.0105342|
|C  |    0|   15| 2373|    8|    0|   0.0095993|
|D  |    0|    1|   21| 2227|    3|   0.0111012|
|E  |    0|    3|    4|    3| 2515|   0.0039604|


