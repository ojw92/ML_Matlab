function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. 
%

% Selected values of lambda
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% Return training errors in error_train and validation errors in error_val


% We use the different values of lambda for learning theta, but
% recall that when computing errors, we set lambda = 0 
for i = 1:length(lambda_vec)
    theta = trainLinearReg(X, y, lambda_vec(i))
    error_train(i) = linearRegCostFunction(X, y, theta, 0);
    error_val(i) = linearRegCostFunction(Xval, yval, theta, 0);
end
% We will find the lambda for which our cross validation error was minimum
% with lambda_vec(error_val == min(error_val)) and use this optimized lambda
% value found from cross validation error to calculate test error


end
