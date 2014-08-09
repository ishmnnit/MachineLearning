
function [C_t] = LogisticReg (Xtrain,Ytrain,Xtest,Ytest)

fprintf('\n ---- Logistic Regression Classifier ---- \n ')

tic
% Train the classifier
% Train the classifier
glm = GeneralizedLinearModel.fit(Xtrain,double(Ytrain)-1);

% Make a prediction for the test set
Y_t = glm.predict(Xtest);
Y_t= round(Y_t)+1;
[C_t]= Missclassification(Ytest,Y_t);

