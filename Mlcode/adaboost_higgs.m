 %% Initialization
  clear ; close all; clc% In this example, we will hold 30% of the data, selected randomly, for
  
data = load('../higgs/sample2.dat');
data = data(:, 1:end) ;

X = data(:, 2:end);
Y = data(:, 1:1);


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

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 200 , 'Tree', 'type','classification')
Y_t  = ens.predict(Xtest);

figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');

% TestSet Accuracy
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);


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

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 200 , 'Tree', 'type','classification')
Y_t  = ens.predict(Xtest);

figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');

% TestSet Accuracy
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

ens  = fitensemble(Xtrain, Ytrain, 'Bag', 300 , 'Tree', 'type','classification')
Y_t  = ens.predict(Xtest);

figure;
plot(loss(ens, Xtest,Ytest,'mode','cumulative'), 'b');

% TestSet Accuracy
fprintf('Accuracy: %f\t', mean(double(Ytest == Y_t)) * 100);


