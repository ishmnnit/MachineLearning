 %% Initialization
  clear ; close all; clc% In this example, we will hold 40% of the data, selected randomly, for
  
data = load('../test2.dat');
data = data(:, 1:end );

X = data(:, 1:end-1 );
Y = data(:, end);
  
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 30  training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% test phase.
cv = cvpartition(length(data),'holdout', 0.70);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

tic
% Train the classifier

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 1000 , 'Tree', 'type','classification')
%ens1 = fitensemble(Xtrain, Ytrain, 'AdaboostM2', 500 , 'Tree' )
%ens2 = fitensemble(Xtrain, Ytrain, 'subspace', 500 , 'Knn' )
toc

% Make a prediction for the test set
figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');
%{
hold on
plot(loss(ens1, Xtest,Ytest,'mode','cumulative'), 'r');
hold on
plot(loss(ens2, Xtest,Ytest,'mode','cumulative'), 'g');
legend('Bag', 'AdaBoostM1', 'Subspace' );
%}
xlabel('Number of trees');
ylabel('Test classification error');

% TestSet Accuracy
Y_t  = ens.predict(Xtest);
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);

% Compute the confusion matrix
%C_t = confusionmat(Ytest,Y_t);
% Examine the confusion matrix for each class as a percentage of the true class
%C_t = bsxfun(@rdivide,C_t,sum(C_t,2)) * 100

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 50  training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% test phase.
cv = cvpartition(length(data),'holdout', 0.50);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

tic
% Train the classifier

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 1000 , 'Tree', 'type','classification')
%ens1 = fitensemble(Xtrain, Ytrain, 'AdaboostM2', 500 , 'Tree' )
%ens2 = fitensemble(Xtrain, Ytrain, 'subspace', 500 , 'Knn' )
toc

% Make a prediction for the test set
figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');
%{
hold on
plot(loss(ens1, Xtest,Ytest,'mode','cumulative'), 'r');
hold on
plot(loss(ens2, Xtest,Ytest,'mode','cumulative'), 'g');
legend('Bag', 'AdaBoostM1', 'Subspace');
%}
xlabel('Number of trees');
ylabel('Test classification error');

% TestSet Accuracy
Y_t  = ens.predict(Xtest);
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 70  training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% test phase.
cv = cvpartition(length(data),'holdout', 0.30);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

tic
% Train the classifier

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 1000 , 'Tree', 'type','classification')
%ens1 = fitensemble(Xtrain, Ytrain, 'AdaboostM2', 500 , 'Tree' )
%ens2 = fitensemble(Xtrain, Ytrain, 'subspace', 500 , 'Knn' )
toc

% Make a prediction for the test set
figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');
%{
hold on
plot(loss(ens1, Xtest,Ytest,'mode','cumulative'), 'r');
hold on
plot(loss(ens2, Xtest,Ytest,'mode','cumulative'), 'g');
legend('Bag', 'AdaBoostM1', 'Subspace' );
%}
xlabel('Number of trees');
ylabel('Test classification error');

% TestSet Accuracy
Y_t  = ens.predict(Xtest);
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);

