
function NeuralNet(Xtrain,Ytrain,Xtest,Ytest,flag)
[Ytrain,Ytest] = DeepLearningNorm(Ytrain,Ytest,flag);

Xtrain = double(Xtrain) / 255;
Ytrain  = double(Ytrain) ;
Xtest = double(Xtest)/255;
Ytest  = double(Ytest);


% normalize
[Xtrain, mu, sigma] = zscore(Xtrain);
Xtest = normalize(Xtest, mu, sigma);

%% ex1 vanilla neural net
rand('state',0)
error=1.0;
C=ones(1,4);
B=ones(1,4);

n=1;
for i=4:4
    A=ones(1,i);
    for j=3:6
        A=A*j;
        A(1,1)=size(Xtrain,2);
        A(1,i)=size(Ytrain,2);
        nn = nnsetup(A);
        opts.numepochs =  100;   %  Number of full sweeps through data
        opts.batchsize = 100;  %  Take a mean gradient step over this many sample
        [nn, L] = nntrain(nn, Xtrain, Ytrain, opts);
        [er, bad] = nntest(nn,Xtest,Ytest);
        if error > er
            error=er;
            NumHiddenLayers=i;
            NumHiddenunit=j;
            Numepochs=10;
        end
        C(1,n)=(1-er)*100;
        B(1,n)=j;
        n=n+1;
    end
end


plot(B,C);
xlabel('Number of Hidden Layers');
ylabel('Accuracy');
title('Multi Layer Perceptron');
grid on;

fprintf('\n Accuracy=%f\n', (1-error)*100);
fprintf('\n NumHiddenLayes=%f\n',NumHiddenLayers);
fprintf('\n NumHiddenunit=%f\n',NumHiddenunit);
fprintf('\n Numepochs=%f\n',Numepochs);


%% ex2 neural net with L2 weight decay
rand('state',0)
nn = nnsetup([size(Xtrain,2) 100 size(Ytrain,2)]);

nn.weightPenaltyL2 = 1e-4;  %  L2 weight decay
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, Xtrain, Ytrain, opts);

[er, bad] = nntest(nn, Xtest, Ytest);
er


%% ex3 neural net with dropout
rand('state',0)
nn = nnsetup([size(Xtrain,2) 100 size(Ytrain,2)]);

nn.dropoutFraction = 0.5;   %  Dropout fraction 
opts.numepochs =  1;        %  Number of full sweeps through data
opts.batchsize = 100;       %  Take a mean gradient step over this many samples

nn = nntrain(nn, Xtrain, Ytrain, opts);

[er, bad] = nntest(nn, Xtest, Ytest);
er


%% ex4 neural net with sigmoid activation function
rand('state',0)
nn = nnsetup([size(Xtrain,2) 100 size(Ytrain,2)]);

nn.activation_function = 'sigm';    %  Sigmoid activation function
nn.learningRate = 1;                %  Sigm require a lower learning rate
opts.numepochs =  1;                %  Number of full sweeps through data
opts.batchsize = 100;               %  Take a mean gradient step over this many samples

nn = nntrain(nn, Xtrain, Ytrain, opts);

[er, bad] = nntest(nn, Xtest, Ytest);
er

%% ex5 plotting functionality
rand('state',0)
nn = nnsetup([size(Xtrain,2) 20 size(Ytrain,2)]);
opts.numepochs         = 5;            %  Number of full sweeps through data
nn.output              = 'softmax';    %  use softmax output
opts.batchsize         = 1000;         %  Take a mean gradient step over this many samples
opts.plot              = 1;            %  enable plotting

nn = nntrain(nn, Xtrain, Ytrain, opts);

[er, bad] = nntest(nn, Xtest, Ytest);
fprintf('\n---Accuracy=%f\n',double(1-er)*100);
fprintf('\n \n');

