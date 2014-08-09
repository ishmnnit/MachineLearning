function [C_t] = Svm_Matlab (Xtrain,Ytrain,Xtest,Ytest)

fprintf('\n ----- Support Vector Machine ---- \n');
% Linear Kernel
SVMStruct = svmtrain(Xtrain, Ytrain, 'kernel_function', 'linear');
Y_t = svmclassify(SVMStruct,Xtest);

[C_t]=Missclassification(Ytest,Y_t);