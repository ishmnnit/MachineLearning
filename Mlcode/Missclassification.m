function [C_t] = Missclassification(Ytest,Y_t)

%stats =confusionmatStats(Ytest,Y_t);

%{
% TestSet Accuracy
%fprintf('stats.accuracy = (TP + TN)/(TP + FP + FN + TN)') ;
%stats.accuracy

%Precision
fprintf('precision=TP/(TP + FP) for each class label');
stats.precision

% Senstivity
fprintf ('Recall=Sensitivity = TP / (TP + FN) for each class label');
stats.sensitivity

%Specificty
fprintf('stats.specificity = TN / (FP + TN) for each class label');
stats.specificity

%F-Score
fprintf('stats.Fscore = 2*TP /(2*TP + FP + FN) for each class');
stats.Fscore
%}

% TestSet Accuracy
size(Ytest)
size(Y_t)
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);

%{

%Root Mean Square Error
error = Ytest - Y_t;
squareError = error.^2;
meanSquareError = mean(squareError);
rootMeanSquareError = sqrt(meanSquareError);
fprintf('\n Root Mean Square Error : %f \n',rootMeanSquareError);

% Compute the confusion matrix
fprintf('\nConfusion Matrix');

C_t = confusionmat(Ytest,Y_t);
% Examine the confusion matrix for each class as a percentage of the true class
C_t = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100
%}
C_t = confusionmat(Ytest,Y_t);