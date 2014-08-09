function [Ytrain,Ytest]= DeepLearningNorm(Ytrain,Ytest,flag)

train_y=zeros(size(Ytrain,1),2);
test_y=zeros(size(Ytest,1),2);

if flag == 1
    
    for i=1:size(Ytrain,1) 
        if Ytrain(i) == 2
            train_y(i,1)=1;
        else
            train_y(i,2)=1;
        end
    end
    
    for i=1:size(Ytest,1)
        if Ytest(i) == 2
            test_y(i,1)=1;
        else
            test_y(i,2)=1;
        end
    end
    
end
 
if flag ==2
    
    for i=1:size(Ytrain,1)
        train_y(i,Ytrain(i)+1)= 1;
    end
    
    for i=1:size(Ytest,1)
        test_y(i,Ytest(i)+1)= 1;
    end
end

Ytrain=train_y;
Ytest=test_y;

end
