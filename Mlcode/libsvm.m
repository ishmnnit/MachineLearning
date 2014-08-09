function [C_t] = libsvm(Xtrain,Ytrain,Xtest,Ytest)

fprintf('\n ----- Support Vector Machine ---- \n');
% Linear Kernel
model_linear = libsvmtrain(Ytrain, Xtrain, '-t 0');
[Y_t, accuracy_L, dec_values_L] = libsvmpredict(Ytest, Xtest, model_linear);

[C_t]=Missclassification(Ytest,Y_t);