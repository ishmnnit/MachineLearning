function svm_distance = svmdistance(SVMStruct,sample)

% shift and scale the data if necessary
if ~isempty(SVMStruct.ScaleData)
    for c = 1:size(sample, 2)
        sample(:,c) = SVMStruct.ScaleData.scaleFactor(c) * ...
            (sample(:,c) +  SVMStruct.ScaleData.shift(c));
    end
end

% read the SVMStruct
sv = SVMStruct.SupportVectors;
alphaHat = SVMStruct.Alpha;
bias = SVMStruct.Bias;
kfun = SVMStruct.KernelFunction;
kfunargs = SVMStruct.KernelFunctionArgs;

% calculate the distance to the SVM dicision boundry
svm_distance = (feval(kfun,sv,sample,kfunargs{:})'*alphaHat(:)) + bias;

% the sign of distances
groupnames = SVMStruct.GroupNames;
[~,groupString,glevels] = grp2idx(groupnames);

if glevels(1) == 0
    svm_distance = -1*svm_distance;
end