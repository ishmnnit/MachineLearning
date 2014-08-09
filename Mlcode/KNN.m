function [C_t] = KNN (Xtrain,Ytrain,Xtest,Ytest)
fprintf('\n ---- K Nearest Neighbour Classifier ---- \n ')
% Train the classifier
knn = ClassificationKNN.fit(Xtrain,Ytrain,'Distance','seuclidean');

% Make a prediction for the test set
Y_t = knn.predict(Xtest);

[C_t]=Missclassification(Ytest,Y_t);

