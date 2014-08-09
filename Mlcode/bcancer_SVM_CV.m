clear ; close all; clc
% In this file, we will test SVM using code in matlab itself

data = load('bcancer_filter.txt');

X = data(:,2:end-1);
Y = data(:,end);

% Change class labels to 1 and 0
Y(Y==2) = 1;
Y(Y==4) = 0;

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

% SVM training by Linear Kernel
ACC_linear = zeros(5,1);
ACC_linear_best = 0;
C_linear_best = 0;
for j = 1:5
    C = 10^(j-3);   % C varies from 10^(-2) to 10^2
    for i = 1:5
        cvtest = (indices == i); cvtrain = ~cvtest;
        SVMStruct = svmtrain(Xtrain(cvtrain,:), Ytrain(cvtrain,:), ...
            'kernel_function', 'linear','boxconstraint',C, ...
            'method','LS');
        Ytrain_cv = svmclassify(SVMStruct, Xtrain(cvtest,:));
        ACC = Evaluate_acc(Ytrain(cvtest,:),Ytrain_cv);
        ACC_linear(j) = ACC_linear(j) + ACC/5;
    end
    if ACC_linear(j) > ACC_linear_best
        ACC_linear_best = ACC_linear(j);
        C_linear_best = C;
    end
end
ACC_linear
ACC_linear_best
C_linear_best


% SVM training by Gaussian Kernel
ACC_rbf = zeros(5,5);
ACC_rbf_best = 0;
C_rbf_best = 0;
sigma_rbf_best = 0;
for j = 1:5
    for k = 1:5
        C = 10^(j-3);   % C varies from 10^(-2) to 10^2
        sigma = 10^(k-3);   % sigma varies from 10^(-2) to 10^2
        for i = 1:5
            cvtest = (indices == i); cvtrain = ~cvtest;
            SVMStruct = svmtrain(Xtrain(cvtrain,:), Ytrain(cvtrain,:), ...
                'kernel_function', 'rbf', 'boxconstraint', C, ...
                'rbf_sigma',sigma, 'method','LS');
            Ytrain_cv = svmclassify(SVMStruct, Xtrain(cvtest,:));
            ACC = Evaluate_acc(Ytrain(cvtest,:),Ytrain_cv);
            ACC_rbf(j,k) = ACC_rbf(j,k) + ACC/5;
        end
        if ACC_rbf(j,k) > ACC_rbf_best
            ACC_rbf_best = ACC_rbf(j,k);
            C_rbf_best = C;
            sigma_rbf_best = sigma;
        end
    end
end
ACC_rbf
ACC_rbf_best
C_rbf_best
sigma_rbf_best


if ACC_linear_best >= ACC_rbf_best
    % Get the accuracy on test data by Linear Kernel
    kernel = 'linear'
    C_linear_best
    SVMStruct = svmtrain(Xtrain, Ytrain, 'kernel_function', 'linear', ...
        'boxconstraint', C_linear_best, 'method','LS');
    Ytest_pred = svmclassify(SVMStruct,Xtest);
    ACC = Evaluate_acc(Ytest,Ytest_pred)
else
    % Get the accuracy on test data by Gaussian Kernel
    kernel = 'Gaussian'
    C_rbf_best
    sigma_rbf_best
    SVMStruct = svmtrain(Xtrain, Ytrain, 'kernel_function', 'rbf', ...
        'boxconstraint', C_rbf_best, 'rbf_sigma',sigma_rbf_best, ...
        'method','LS');
    Ytest_pred = svmclassify(SVMStruct,Xtest);
    ACC = Evaluate_acc(Ytest,Ytest_pred)
end
