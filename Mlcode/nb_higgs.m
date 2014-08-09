%% Initialization
clear ; close all; clc% In this example, we will hold 40% of the data, selected randomly, for
  
data = load('../higgs/sample7.dat');
data = data(:, 1:end) ;

X = data(:, 2:end);
Y = data(:, 1:1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 30 training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
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

Nb  = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'normal');
Nb1 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'kernel');
%Nb2 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mn');
Nb3 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mvmn');

% TestSet Accuracy
Y_t = Nb.predict(Xtest);
Y_t1 = Nb1.predict(Xtest);
%Y_t2 = Nb2.predict(Xtest);
Y_t3 = Nb3.predict(Xtest);

fprintf('Accuracy: normal 30 : %f\n', mean(double(Ytest == Y_t)) * 100);
fprintf('Accuracy: kernel 30 : %f\n', mean(double(Ytest == Y_t1)) * 100);
%fprintf('Accuracy: mn 30 : %f\n', mean(double(Ytest == Y_t2)) * 100);
fprintf('Accuracy: mvmn 30 : %f\n', mean(double(Ytest == Y_t3)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 50 training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
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

Nb  = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'normal');
Nb1 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'kernel');
%Nb2 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mn');
Nb3 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mvmn');

% TestSet Accuracy
Y_t = Nb.predict(Xtest);
Y_t1 = Nb1.predict(Xtest);
%Y_t2 = Nb2.predict(Xtest);
Y_t3 = Nb3.predict(Xtest);

fprintf('Accuracy: normal 50 : %f\n', mean(double(Ytest == Y_t)) * 100);
fprintf('Accuracy: kernel 50 : %f\n', mean(double(Ytest == Y_t1)) * 100);
%fprintf('Accuracy: mn 50 : %f\n', mean(double(Ytest == Y_t2)) * 100);
fprintf('Accuracy: mvmn 50 : %f\n', mean(double(Ytest == Y_t3)) * 100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 70 training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
  
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

Nb  = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'normal');
Nb1 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'kernel');
%Nb2 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mn');
Nb3 = NaiveBayes.fit(Xtrain, Ytrain, 'dist', 'mvmn');

% TestSet Accuracy
Y_t = Nb.predict(Xtest);
Y_t1 = Nb1.predict(Xtest);
%Y_t2 = Nb2.predict(Xtest);
Y_t3 = Nb3.predict(Xtest);

fprintf('Accuracy: normal 30 : %f\n', mean(double(Ytest == Y_t)) * 100);
fprintf('Accuracy: kernel 30 : %f\n', mean(double(Ytest == Y_t1)) * 100);
%fprintf('Accuracy: mn 30 : %f\n', mean(double(Ytest == Y_t2)) * 100);
fprintf('Accuracy: mvmn 30 : %f\n', mean(double(Ytest == Y_t3)) * 100);

