clear ; close all; clc

% Load the data
data = load('bcancer_filter.txt');

X = data(:,2:end-1);
Y = data(:,end);

% 30% Training data, 70% Testing data
cv = cvpartition(length(data),'holdout',.7);
% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

% 5-fold cross validation within training data
indices = crossvalind('Kfold',Ytrain,5);

% Algorithm training by method1 with parameter1
ACC_method1 = zeros(n,1);   % change parameter1 n times
ACC_method1_best = 0;
parameter1_best = 0;
for j = 1:n
    parameter1 = f(j);   % define the value for paramter1
    for i = 1:5
        cvtest = (indices == i); cvtrain = ~cvtest;
        classifier = train(Xtrain(cvtrain,:), Ytrain(cvtrain,:),method1,parameter1); % train with 4fold
        Ytrain_cv = classify(Xtrain(cvtest,:), classifier); % test with 1fold
        ACC = Evaluate_acc(Ytrain(cvtest,:),Ytrain_cv); % acc for this run
        ACC_method1(j) = ACC_method1(j) + ACC/5;  % average for the 5 runs
    end
    if ACC_method1(j) > ACC_method1_best
        ACC_method1_best = ACC_method1(j);
        parameter1_best = parameter1;
    end
end
ACC_method1
ACC_method1_best
parameter1_best


% Algorithm training by method2 with both parameter2 & parameter3
ACC_method2 = zeros(n,n);    % change parameter2 & parameter3 n*n times
ACC_method2_best = 0;
parameter2_best = 0;
parameter3_best = 0;
for j = 1:n
    for k = 1:n
        parameter2 = f(j); 
        parameter3 = g(j);
        for i = 1:5
            cvtest = (indices == i); cvtrain = ~cvtest;
            classifier = train(Xtrain(cvtrain,:), Ytrain(cvtrain,:),method2,parameter2,parameter3); % train with 4fold
            Ytrain_cv = classify(Xtrain(cvtest,:), classifier); % test with 1fold
            ACC = Evaluate_acc(Ytrain(cvtest,:),Ytrain_cv); % acc for this run
            ACC_method1(j) = ACC_method1(j) + ACC/5;  % average for the 5 runs
        end
        if ACC_method2(j,k) > ACC_method2_best
            ACC_method2_best = ACC_method2(j,k);
            parameter2_best = parameter2;
            parameter3_best = parameter3;
        end
    end
end
ACC_method2
ACC_method2_best
parameter2_best
parameter3_best


if ACC_method1_best >= ACC_method2_best
    % Get the accuracy on test data by method1
    method = 'method1'
    parameter1_best
    classifier = train(Xtrain, Ytrain,method1,parameter1); % train with the whole training set
    Ytest_pred = classify(Xtest, classifier); % test with the testing set
    ACC = Evaluate_acc(Ytest,Ytest_pred); % acc for the testing set
    
else
    % Get the accuracy on test data by method2
    method = 'method2'
    parameter2_best
    parameter3_best
    classifier = train(Xtrain, Ytrain, method2,parameter2,parameter3); % train with the whole training set
    Ytest_pred = classify(Xtest, classifier); % test with the testing set
    ACC = Evaluate_acc(Ytest,Ytest_pred); % acc for the testing set
    
end
