 %% Initialization
  clear ; close all;
  
  fprintf('\n  <= Choose Dataset to load =>  \n');
  fprintf('Press 1 for Breast Cancer Dataset\n');
  fprintf('Press 2 Optdigit Dataset\n');
  fprintf('Press 3 Higgs Boson Dataset\n');
  
  dataset=input('\n<= Input Dataset Number => \n');
  
  switch dataset
      case 1
          data = load('bcancer_filter.txt');
      
      case 2
          data= load('optdigits.tra');
      
      case 3
          data=load('higgs/sample2.dat');
 
      otherwise
            fprintf('Wrong Input , Unknown Action');
          
  end


  if  dataset == 3 
      X = data(:,2:end);
      Y = data(:,1);
  else
      X = data(:,2:end-1);
      Y=  data(:,end);
  end
      

% Data Visualization

N = size(X, 1);
m = mean(X); % each row is a data sample
data_m = X - repmat(m, N, 1);
covar = data_m'*data_m/N; % or N-1 for unbiased estimate
[U,S,V] = svd(covar);
reduced_data = data_m*V(:,1:2); % reduce to 2 components
gscatter(reduced_data(:,1),reduced_data(:,2),Y);
xlabel('Principal Feature1');
ylabel('Principal Feature2');
title('Dataset Visualization');
grid on;


% test phase.
cv = cvpartition(length(data),'holdout',.5);

% Training set
Xtrain = X(training(cv),:);
Ytrain = Y(training(cv),:);

% Test set
Xtest = X(test(cv),:);
Ytest = Y(test(cv),:);




fprintf('\n  <= Choose Your Classifier =>  \n');
fprintf('Press 1 Deep Belief Network \n');
fprintf('Press 2 Stacked Auto Encoders \n');
fprintf('Press 3 Multi Layer Perceptron \n');
fprintf('Press 4 Random Forest \n');
fprintf('Press 5 Bagged Decision Tree \n');
fprintf('Press 6 Support Vector Machine \n');
fprintf('Press 7 Decision Tree Classification \n');
fprintf('Press 8 K-Nearest Neighbours \n');
fprintf('Press 9 Neural Network Classifier \n');
fprintf('Press 10 Logistic Regresion \n');
fprintf('Press 11 Naive Bayes Classifier \n');
fprintf('\n PRESS 0 FOR EXIT \n')


while(true)
    
    ClassifierType=input('\n\n Input ClassifierType\n\n');
    if ClassifierType == 0
        break;
    end
    
    switch(ClassifierType)
        
        case 1
            % Deep Belief Network
            DBN(Xtrain,Ytrain,Xtest,Ytest,dataset);
        case 2
            % Stacked Auto-Encoders
            SAE(Xtrain,Ytrain,Xtest,Ytest,dataset);      
        case 3
            %Neural Network  Classifier
            NeuralNet(Xtrain,Ytrain,Xtest,Ytest,dataset);
            
        case 4
            % Random Forest
            [RF_t] = RandomForest(Xtrain,Ytrain,Xtest,Ytest);
        case 5
            %Bagged Decision Tree
            [TB_t] =BaggedDt(Xtrain,Ytrain,Xtest,Ytest);
            
        case 6
            %Support Vector Machine Classification
            [S_t] = Svm_Matlab (Xtrain,Ytrain,Xtest,Ytest);
            
        
        case 7   
           %Decision Tree Classification
           % test phase.
           [D_t] = DecisionTree (Xtrain,Ytrain,Xtest,Ytest);      
        case 8
            % K nearest Neighbour Classifier
            [K_t]= KNN(Xtrain,Ytrain,Xtest,Ytest);
        case 9
            %Logistic Regression Classifier
            [L_t]=LogisticReg(Xtrain,Ytrain,Xtest,Ytest);   
        case 10
            %Discrminant Analysis Classifier
            [DA_t] = DiscrimAnalysis(Xtrain,Ytrain,Xtest,Ytest); 
        case 11    
            %NaiveBayes Classifier 
            [NB_t] = NaiveBayes_M(Xtrain,Ytrain,Xtest,Ytest); 
            
        otherwise
            fprintf('Wrong Input , Unknown Action');
   end
end