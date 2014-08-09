function [C_t] = DecisionTree (Xtrain,Ytrain,Xtest,Ytest)
% Train the classifier
t = ClassificationTree.fit(Xtrain,Ytrain);

% Make a prediction for the test set

Y_t = t.predict(Xtest);

fprintf('\n ---- Decision Tree ---- \n ')
[C_t]=Missclassification(Ytest,Y_t);