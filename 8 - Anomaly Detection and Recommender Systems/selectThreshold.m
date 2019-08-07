function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % Compute the F1 score of choosing epsilon as the
    % threshold and place the value in F1. 

    % Our prediction is that if probability of our data point is less than
    % epsilon, then it is classified as an anomaly
    predictions = (pval < epsilon);
    % true positive, false positive, false negative
    % predictions == yval checks for tp & tn, but taking its sum won't help
    % since tn will be represented as 0 and will not be counted towards the sum
    tp = sum((predictions == 1) & (yval == 1));
    fp = sum((predictions == 1) & (yval == 0));
    fn = sum((predictions == 0) & (yval == 1));
    % An alternative way to do this is by counting 1's and 0's
    % comparison = predictions == yval;
    % tp = sum(comparison == 1);
    % tn = sum(comparison == 0);
    
    % precision = # of true positive / (true positive + false positive)
    prec = tp / (tp + fp);
    
    % recall = # of true positive / (true positive + false negative)
    rec = tp / (tp + fn);
    
    F1 = 2 * prec * rec / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
