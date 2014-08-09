clear ; close all; clc
% In this example, we will hold 30% of the data, selected randomly, for
% SVM using code in libsvm

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
model_linear = svmtrain(Ytrain, Xtrain, '-t 0');
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(Ytest, Xtest, model_linear);

accuracy_L % Display the accuracy using linear kernel


