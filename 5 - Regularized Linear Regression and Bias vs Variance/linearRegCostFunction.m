function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost and gradient of regularized linear 
% regression for a particular choice of theta.


h = X * theta;
J = 0.5/m * (h - y)' * (h - y) + lambda/2/m * (theta(2:end,:)' * theta(2:end,:));
grad = 1/m * (X'*(h - y) + lambda * [0; theta(2:end)]);


grad = grad(:);

end
