function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z.

g = zeros(size(z));

% Compute the gradient of the sigmoid function evaluated at
% each value of z (z can be a matrix, vector or scalar).


g = 1 ./ (1+exp(-z)) .* (1 - 1 ./ (1+exp(-z)));



end
