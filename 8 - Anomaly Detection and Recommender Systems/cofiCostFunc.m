function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%   The input params is the unrolled form of X & Theta, going down
%   one column at a time

% Unfold the U and W matrices from params
% reshape("all of params", "# of rows", "# of columns")
% Undoes param = (X(:), Theta(:))
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% Compute the cost function and gradient for collaborative filtering 
% First implement the cost function without regularization a
% Then implement the gradient and use the checkCostFunction routine to check
% that the gradient is correct. Finally, implement regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%

% X * Theta' gives num_movies x num_users matrix
h = X * Theta';
% We don't want to calculate the cost movies that have no ratings, so
% multiply by R to make elements of h = 0 where there is no rating
% Y already has 0's where there is no rating, so no change needed
h = h .* R;
J = 1/2* (sum(sum((h - Y).^2)) + ...
    lambda* (sum(sum(Theta .^2)) + sum(sum(X .^2)) ) );

X_grad = (h - Y) * Theta + lambda * X; % size is num_movies x num_features

Theta_grad = (h - Y)' * X + lambda * Theta; % size is num_users x num_features


grad = [X_grad(:); Theta_grad(:)];

end
