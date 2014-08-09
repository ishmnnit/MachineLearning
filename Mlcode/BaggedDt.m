% Cost of misclassification
function [Y_t] = BaggedDt(Xtrain,Ytrain,Xtest,Ytest)

fprintf('\n Bagged Decision Tree \n ');
tb = TreeBagger(150,Xtrain,Ytrain,'method','classification');

% Make a prediction for the test set
Y_t = tb.predict(Xtest);
[C_t]= Missclassification(Ytest,Y_t);
