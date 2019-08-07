function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. Return the optimal C and sigma based on a cross-validation set
%

C = 1;
sigma = 0.3;

% Return optimal C and sigma learning parameters found using cross validation set
%

% Trying out all 8^2 combinations of C and sigma values
Cv = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
Sv = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
Pred = zeros(size(Cv,1));

for i = 1:length(Cv)
    for j = 1:length(Sv)
        model = svmTrain(X, y, Cv(i), @(x1, x2) gaussianKernel(x1, x2, Sv(j)));
        
        % predict labels using our trained SVM model
        predictions = svmPredict(model, Xval);
        % this next line shows the accuracy of our prediction
        % sum of all wrong examples labeled 1, then divide by total number of examples 
        Pred(i,j) = mean(double(predictions ~= yval));
    end
end

% Minimum error percentage Min and its index Ind_i, Ind_j
Min = min(min(Pred));
[Ind_i Ind_j] = find(Pred == Min);
C = Cv(Ind_i);
sigma = Sv(Ind_j);



end
