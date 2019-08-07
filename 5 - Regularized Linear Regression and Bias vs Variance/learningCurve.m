function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, should do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% Return training errors in error_train and cross validation errors in error_val


% This computes error of examples when training up to i examples
% When i=1, train 1 ex. When i=2, train 2 ex, and so on.
% We will train m examples for both training & validation sets
% but compute error for cross validation on the entire set each time
% Both this method and the linearRegCostFunction method work fine
% for i = 1:m
%     Theta = trainLinearReg(X(1:i,:), y(1:i), lambda);
%     error_train(i) = 0.5/m * (X(1:i,:)*Theta - y(1:i))' * (X(1:i,:)*Theta - y(1:i));
%     error_val(i) = 0.5/m * (Xval(:,:)*Theta - yval(:))' * (Xval(:,:)*Theta - yval(:)) ...
%         + 0/2/m * (Theta(2:end,:)' * Theta(2:end,:));
% end

% or make use of linearRegCostFunction:
for i = 1:m
    Theta = trainLinearReg(X(1:i,:), y(1:i), lambda);
    error_train(i) = linearRegCostFunction(X(1:i,:), y(1:i), Theta, 0);
    error_val(i) = linearRegCostFunction(Xval(:,:), yval(:), Theta, 0);
end


end
