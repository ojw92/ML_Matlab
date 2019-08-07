function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta
% Note: grad should have the same dimensions as theta


% If y = 0, θTx must be very negative for sigmoid to be 0 and cost to be 0
% If y = 1, θTx must be very positive for sigmoid to be 1 and cost to be 0


J = 1/m * (-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta)));

for i = 1:size(theta)
    temp = 1/m * sum(X(:,i)'*(sigmoid(X*theta) - y));
    grad(i) = temp;
end

grad;

end
