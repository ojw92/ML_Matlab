function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% Compute the mean of the data and the variances
%


mu = mean(X, 1)'; % gives mean of each column (gives a row vector)
sigma2 = 1/m * sum((X - mu').^2, 1); % sum of each column (row vector)


end
