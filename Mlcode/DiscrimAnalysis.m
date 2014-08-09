function [C_t] = DiscrimAnalysis(Xtrain,Ytrain,Xtest,Ytest)

fprintf('\n ---- Discrminant Analysis Classifier ----- \n');
% Train the classifier
da = ClassificationDiscriminant.fit(Xtrain,Ytrain,'discrimType','quadratic');

% Make a prediction for the test set
Y_t = da.predict(Xtest);

[C_t]= Missclassification(Ytest,Y_t);


