function [ C_t] = RandomForest( Xtrain,Ytrain,Xtest,Ytest)
%   Summary of this function goes here
%   Detailed explanation go
%      ntrees        : number of trees in the ensemble (default 50)
%
%      oobe         : out-of-bag error calculation, 
%                      values ('y'/'n' -> yes/no) (default 'n')
%
%      nsamtosample : number of randomly selected (with
%                      replacement) samples to use to grow
%                      each tree (default num_samples)
%      method       : the criterion used for splitting the nodes
%                           'g' : gini impurity index (classification)
%                           'c' : information gain (classification)
%                            'r' : squared error (regression)
%       minparent    : the minimum amount of samples in an impure node
%                      for it to be considered for splitting
%
%       minleaf      : the minimum amount of samples in a leaf
%
%       weights      : a vector of values which weigh the samples 
%                      when considering a split
%
%       nvartosample : the number of (randomly selected) variables 
%                      to consider at each node 
methods='gcr';
error=1.0;
A=ones(1,20);
B=ones(1,20);
n=1;

for i=15:20
    fprintf('\nNvartosaple=%d\t',i);
    for j=1:1
        %fprintf('%s',methods(j));
        k=200;
        while(k<=440)
            fprintf('\tk=%d\t',k);
            fprintf('%s',methods(j));
            size(Xtrain)
            size(Ytrain)
            size(Xtest)
            model = train_RF(double(Xtrain),double(Ytrain),'ntrees', k,'oobe','y','nsamtosample',20,'method',methods(j),'nvartosample',i);
            Y_t = eval_RF(double(Xtest), model, 'oobe', 'y');
  
  
            [C_t]=Missclassification(Ytest,Y_t);
            er= mean(double(Ytest ~= Y_t));
            if error > er
                error=er;
                nvartosample=i;
                method=j;
                numTrees=k;
            end
            A(1,n)=(1-er)*100;
            B(1,n)=n;
            n=n+1;
           
            if k==2
                k=3;
            else
                k=k+20;
            end
        end   
    end
end

fprintf('\n Accuracy=%f',(1-error)*100);
fprintf('\n NumofRandomly selected Variable=%f',nvartosample);
fprintf('\n Method=%f',method);
fprintf('\n numTress=%f',numTrees);
plot(A);
xlabel('Number of Randomly Selected Variable');
ylabel('Accuracy');
title('Random Forest');
grid on;
end
