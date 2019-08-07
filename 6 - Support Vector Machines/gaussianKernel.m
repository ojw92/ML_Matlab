function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
% Unrolls matrix by going down col 1, then col 2, then col 3, and so on
% Since x1 & x2 were row vectors, there are now features on the rows for each example
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% Return the similarity between x1 and x2 computed using a Gaussian kernel
% with bandwidth sigma


sim = exp(-0.5 * (x1 - x2)'* (x1 - x2) / sigma^2);

    
end
