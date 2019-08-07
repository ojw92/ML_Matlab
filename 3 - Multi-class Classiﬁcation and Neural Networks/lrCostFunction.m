function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta


temp = theta;
temp(1) = 0;
J = 1/m * sum(-y' * log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta)) ) + lambda/2/m *temp'*temp;

grad = 1/m * (X' * (sigmoid(X*theta)-y)) + lambda/m*temp;

% Instead of using temp for regularization, an alternative method would be to use:
% theta(2:length(theta))


grad = grad(:);

end
