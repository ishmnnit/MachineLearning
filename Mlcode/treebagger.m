% Cost of misclassification
function [Y_tb] = treebagger(Xtrain,Ytrain,Xtest,Ytest)

opts = statset('UseParallel',true);
% Train the classifier
% Number of Trees
% B = TreeBagger(NTrees,X,Y,'param1',val1,'param2',val2,...)
% First argument  is Number of Trees

tb = TreeBagger(150,Xtrain,Ytrain,'method','classification','Options',opts,'OOBVarImp','on');

% Make a prediction for the test set
[Y_tb, classifScore] = tb.predict(Xtest);
Y_tb = nominal(Y_tb);
fprintf('Size of Y_tbb');
size(Y_tb)

fprintf('Size of Ytest ');
size(Ytest)

% Compute the confusion matrix
C_tb = confusionmat(Ytest,Y_tb);

% Examine the confusion matrix for each class as a percentage of the true class
C_tb = bsxfun(@rdivide,C_tb,sum(C_tb,2)) * 100