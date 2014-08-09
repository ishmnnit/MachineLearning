function ACC = Evaluate_acc(actual,pred)


% This fucntion calculates the following metrics:
%        - ACC (accuracy)
% 
% 
% Input: actual     = 0-1 binary Column vector with actual class labels
%                     of the testing samples
%        pred       = 0-1 binary Column vector with predicted class labels
%                     of the testing samples


idx = (actual()==1);

p = length(actual(idx));
n = length(actual(~idx));
N = p+n;

tp = sum(actual(idx)==pred(idx));
tn = sum(actual(~idx)==pred(~idx));


ACC = (tp+tn)/N;

