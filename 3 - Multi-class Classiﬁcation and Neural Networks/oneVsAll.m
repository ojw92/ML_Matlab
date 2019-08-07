function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];


% Train num_labels logistic regression classifiers with regularization

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
    [theta] = ...
        fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
        initial_theta, options);
    all_theta(c,:) = theta';
end
all_theta    


% K = num_labels = 10 classifications, n = 400 features of X
% For each of 10 classifications, we find the 400 values of theta
% For this example, think of 1 classification as "effect" of 400 inputs on one
%  neuron, and that "effect" is represented as an output h(x)
% Visualize a neural network with 400 inputs (features of X) and one
%  inner layer a with 400 neurons: a = g(theta^(1)*X), where theta^(1) is 10 x 400
% Each classification corresponds to each row of the matrix a, which is 10 x 1


end
