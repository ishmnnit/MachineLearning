function DBN( Xtrain,Ytrain,Xtest,Ytest,flag)

% First Preprocess Data Before Running
[Ytrain,Ytest] = DeepLearningNorm(Ytrain,Ytest,flag);

Xtrain = double(Xtrain);
Xtest  = double(Xtest); 
Ytrain = double(Ytrain);
Ytest  = double(Ytest);


a = min(Xtrain(:));
b = max(Xtrain(:));
ra = 0.9;
rb = 0.1;
Xtrain = (((ra-rb) * (Xtrain - a)) / (b - a)) + rb;
Xtest = (((ra-rb) * (Xtest - a)) / (b - a)) + rb;




%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN
rng(0,'v5uniform');
%train dbn

error=1.0;

for i=4:5
    A=ones(1,i);
    for j=5:5:10
        dbn.sizes = A*j;
        for k=3:4
            opts.numepochs =   k;
            for l=100:100
                opts.batchsize = l;
                opts.momentum  =   0.2;
                for m=1:1
                    opts.alpha     =   .1*m; % Learning Parameter
                    dbn = dbnsetup(dbn, Xtrain, opts);
                    dbn = dbntrain(dbn, Xtrain, opts);
                    %unfold dbn to nn
                    nn = dbnunfoldtonn(dbn, size(Ytrain,2));
                    nn.activation_function = 'tanh_opt';
                    %train nn
                    opts.numepochs =  100;
                    opts.batchsize = 100;
                    nn = nntrain(nn, Xtrain, Ytrain, opts);
                    [er, bad] = nntest(nn, Xtest, Ytest);
                    fprintf('\n---Accuracy=%f\n',double(1-er)*100);
                    if error > er
                        error=er;
                        NumofHiddenlayer=i;
                        NumofNeuron=j;
                        Numeochs=k;
                        optbatchsize=l;
                        learningParameter=.01*m;                                      
                    end
                end
            end
        end
    end
end
fprintf('\n error=%f\n',error);
fprintf('\n NumofHiddenlayer=%f\n',NumofHiddenlayer);
fprintf('\n NumofNeuron=%f \n',NumofNeuron);
fprint('\n Numeochs=%f \n',Numeochs');
fprintf('\n optbatchsize=%f',optbatchsize);
fprintf('\n learningParameter=%f',learningParameter);

