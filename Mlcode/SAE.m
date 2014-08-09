function SAE(train_x,train_y,test_x,test_y,flag);

%{
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);
size(train_x)
size(train_y)
size(test_x)
size(test_y)

%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
rand('state',0)
sae = saesetup([784 100]);
sae.ae{1}.activation_function       = 'sigm';
sae.ae{1}.learningRate              = 1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')

% Use the SDAE to initialize a FFNN
nn = nnsetup([784 100 10]);
nn.activation_function              = 'sigm';
nn.learningRate                     = 1;
nn.W{1} = sae.ae{1}.W{1};

% Train the FFNN
opts.numepochs =   1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);
end
%}


[train_y,test_y] = DeepLearningNorm(train_y,test_y,flag);

train_x = double(train_x);
test_x  = double(test_x);
train_y = double(train_y);
test_y  = double(test_y);


% normalize
[train_x, mu, sigma] = zscore(train_x);
test_x = normalize(test_x, mu, sigma);


%%  ex1 train a 100 hidden unit SDAE and use it to initialize a FFNN
%  Setup and train a stacked denoising autoencoder (SDAE)
A=ones(1,5);
C=ones(1,5);

rand('state',0);
sae = saesetup([size(train_x,2) size(train_x,2)]);
sae.ae{1}.activation_function       = 'tanh_opt';
sae.ae{1}.learningRate              = .1;
sae.ae{1}.inputZeroMaskedFraction   = 0.5;
opts.numepochs =   1;
opts.batchsize = 100;
sae = saetrain(sae, train_x, opts);
visualize(sae.ae{1}.W{1}(:,2:end)')
error=1.0

% Use the SDAE to initialize a FFNN
for i=10:15
    B=ones(1,i);
    B=B*size(test_x,2);
    B(1,i)=size(test_y,2);
    nn = nnsetup(B);
    nn.activation_function = 'tanh_opt';
    nn.learningRate = .1;
    nn.W{1} = sae.ae{1}.W{1};
    % Train the FFNN
    opts.numepochs =   1;
    opts.batchsize = 100;
    nn = nntrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);
    
    if error > er
        error=er ;      
    end
    
    A(1,i-2)=(1-er)*100;
    C(1,i-2)=i;
    fprintf('\n---Accuracy=%f\n',double(1-er)*100);
    fprintf('\n \n');
    
end


plot(C,A);
xlabel('Number of Hidden Layers');
ylabel('Accuracy');
title('Stacked Auto Encoders');
grid on;

end