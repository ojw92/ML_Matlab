function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

% Compute the cost of a particular choice of theta.
% Compute the partial derivatives and set grad to the partial
% derivatives of the cost w.r.t. each parameter in theta

% fyi, X(:,1)'*X(:,1) = sum(X(:,1).*X(:,1))

J = 1/m*(-y'*log(sigmoid(X*theta)) - (1-y)'*log(1-sigmoid(X*theta))) + lambda/(2*m)*sum(theta(2:size(theta),:).^2);
for i = 1:size(theta)
    if i == 1
        grad(i) = 1/m*X(:,i)'*(sigmoid(X*theta)-y);
    else
        grad(i) = 1/m*X(:,i)'*(sigmoid(X*theta)-y) + lambda/m*theta(i);
    end
end


end
