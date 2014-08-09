clear ; close all; clc
% In this example, we will hold 30% of the data, selected randomly, for
% SVM using code in matlab itself

data = load('bcancer_filter.txt');
data = data(:,2:end) ;

X = data(:,2:end-1);
Y = data(:,end);

% test phase.
cv = cvpartition(length(data),'holdout',.3);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

% Linear Kernel
SVMStruct = svmtrain(Xtrain, Ytrain, 'kernel_function', 'linear');
Y_t = svmclassify(SVMStruct,Xtest);

% Compute the confusion matrix
C_t = confusionmat(Ytest,Y_t);
% Examine the confusion matrix for each class as a percentage of the true class
C_t = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100
