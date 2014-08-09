function [ACC,FSC,LFT,ROC,APR,BEP,RMS,MXE] = Evaluate(actual,pred_value)


% This fucntion calculates the following metrics:
%     Threshold Metrics:
%        - ACC (accuracy)
%        - FSC (F-score)
%        - LFT (lift)
%     Ordering/Rank Metrics:
%        - ROC (area under the ROC curve)
%        - APR (average precision)
%        - BEP (precision/recall break even point)
%     Probability Metrics:
%        - RMS (root mean squared error)
%        - MXE (mean cross entropy)
% 
% 
% Input: actual     = 0-1 binary Column vector with actual class labels
%                     of the testing samples
%        pred_value = Column vector with predicted class values in [0,1] 
%                     by the classification model
% 
% 
% Output: All the performance metrics



% Threshold Metrics:
%    - ACC (accuracy)
%    - FSC (F-score)
%    - LFT (lift)
idx = (actual()==1);

p = length(actual(idx));
n = length(actual(~idx));
N = p+n;

thresh = 0.5;
pred_class = zeros(size(actual));
pred_class(pred_value >= thresh) = 1;

tp = sum(actual(idx)==pred_class(idx));
tn = sum(actual(~idx)==pred_class(~idx));
fp = n-tn;
fn = p-tp;


% ACC (accuracy)
ACC = (tp+tn)/N;


% FSC (F_score)
precision = tp/(tp+fp);
recall = tp/p;
FSC = 2*((precision*recall)/(precision + recall));


% LFT (lift)
[pred_vsorted,IX]=sort(pred_value,'descend');
pred_sorted = pred_class(IX);
actual_sorted = actual(IX);

fixed_percent = 0.25;
nlift = round(N*fixed_percent);

actual_lift = actual_sorted(1:nlift);
pred_lift = pred_sorted(1:nlift);

idx_lift = (actual_lift()==1);
tp_lift = sum(actual_lift(idx_lift)==pred_lift(idx_lift));

LFT = (tp_lift/nlift)/(p/N);



% Ordering/Rank Metrics:
%    - ROC (area under the ROC curve)
%    - APR (average precision)
%    - BEP (precision/recall break even point)

% ROC (area under the ROC curve)
[X,Y,T,AUC] = perfcurve(actual,pred_value,1);
ROC = AUC;


% APR (average precision)
n_t = 101;
thresh_t = zeros(n_t,1);
precision_t = zeros(n_t,1);
recall_t = zeros(n_t,1);

for i = 1:n_t
    
    thresh_t(i) = 1 - (i-1)/(n_t-1);
    
    pred_class_t = zeros(size(actual));
    pred_class_t(pred_value >= thresh_t(i)) = 1;
    
    tp_t = sum(actual(idx)==pred_class_t(idx));
    tn_t = sum(actual(~idx)==pred_class_t(~idx));
    fp_t = n-tn_t;
  
    precision_t(i) = tp_t/(tp_t+fp_t);
    recall_t(i) = tp_t/p;
    
end

APR = 0;
for i = 1:n_t-1
    APR = APR + abs((recall_t(i+1)-recall_t(i))*...
        (precision_t(i+1)+precision_t(i))/2);
end


% BEP (precision/recall break even point)
diff = abs(precision_t-recall_t);
ind_BEP = find(diff == min(diff),1,'first');
BEP = precision_t(ind_BEP);



% Probability Metrics:
%    - RMS (root mean squared error)
%    - MXE (mean cross entropy)

% RMS (root mean squared error)
RMS = sqrt(sum((pred_value-actual).^2)/N);

% MXE (mean cross entropy)
MXE = (-1/N)*sum(actual.*log(pred_value+eps)+...
                 (1-actual).*log(1-pred_value+eps));
