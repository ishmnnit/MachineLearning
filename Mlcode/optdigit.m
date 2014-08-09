%% Initialization
  clear ; close all; clc% In this example, we will hold 40% of the data, selected randomly, for
  
fprintf('Plotting Data ...\n')
data = load('optdigits.tra');
data = data(:,2:end) ;

X = data(:,2:end-1);
Y = data(:,end);

%PlotData
%PlotData(X,Y);
  
% test phase.
cv = cvpartition(length(data),'holdout',.3);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);
% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);

%Feature Selection
%fun = @(Xtrain,Ytrain,Xtest,Ytest)...
 %     (sum(~strcmp(Ytest,classify(Xtest,Xtrain,Ytrain,'quadratic'))));
  
%opts = statset('display','iter');
%[fs,history] = sequentialfs(fun,Xtrain,Ytrain,'cv',cv,'options',opts);

% Random Forest
%[RF_t] = RandomForest(Xtrain,Ytrain,Xtest,Ytest);

% Decision Tree Classification
%[D_t] = DecisionTree (Xtrain,Ytrain,Xtest,Ytest);

%Support Vector Machine Classification
[S_t] = libsvm(Xtrain,Ytrain,Xtest,Ytest);

pause;

% K nearest Neighbour Classifier
%[K_t]= KNN(Xtrain,Ytrain,Xtest,Ytest);

%Neural Network  Classifier
%[NN_t]=NeuralNet(Xtrain,Ytrain,Xtest,Ytest);

%Logistic Regression Classifier
%[L_t]=LogisticReg(Xtrain,Ytrain,Xtest,Ytest);

%Discrminant Analysis Classifier
[DA_t] = DiscrimAnalysis(Xtrain,Ytrain,Xtest,Ytest);

%NaiveBayes Classifier 
[NB_t] = NaiveBayes_M(Xtrain,Ytrain,Xtest,Ytest);

%Bagged Decision Tree
[TB_t] =BaggedDt(Xtrain,Ytrain,Xtest,Ytest);
