function [C_t]= NaiveBayes_M(Xtrain,Ytrain,Xtest,Ytest)
% Train the classifier
fprintf('\n ----- Naive  Bayes Classifier ------ \n ');
Nb = NaiveBayes.fit(Xtrain,Ytrain);

% Make a prediction for the test set
Y_t = Nb.predict(Xtest);

[C_t]= Missclassification(Ytest,Y_t);
