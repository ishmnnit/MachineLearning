clear ; close all; clc
% In this example, we will hold 30% of the data, selected randomly, for
% SVM using code in matlab itself

data = load('bcancer_filter.txt');
data = data(:,2:end) ;

X = data(:,2:end-1);
Y = data(:,end);

% Change class labels to 1 and 0
Y(Y==2) = 1;
Y(Y==4) = 0;

% Training set
cv = cvpartition(length(data),'holdout',.3);
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

% SVM training by Linear Kernel
SVMStruct = svmtrain(Xtrain, Ytrain, 'kernel_function', 'linear');

% Calculate the distance to the SVM dicision boundry
Ytest_dist = svmdistance(SVMStruct,Xtest);



% Not applicable to PLT and ISO
% scale the distance to [0 1]
Ytest_max = max(Ytest_dist);
Ytest_min = min(Ytest_dist);
Ytest_value = zeros(size(Ytest));
for i = 1:size(Ytest)
    if Ytest_dist(i) >= 0
        Ytest_value(i) = 0.5 + 0.5*Ytest_dist(i)/Ytest_max;
    else
        Ytest_value(i) = 0.5 - 0.5*Ytest_dist(i)/Ytest_min;
    end
end


% Calculate the metrics values
% Ytest_value according to scaling or PLT or ISO
[ACC FSC LFT ROC APR BEP RMS MXE] = Evaluate(Ytest,Ytest_value)



