function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples
J = 0;


% Compute the cost of a particular choice of theta


% y_norm = (y - mean(y))/std(y); 
J = 0.5/m*sum((X * theta - y).^2);


end
