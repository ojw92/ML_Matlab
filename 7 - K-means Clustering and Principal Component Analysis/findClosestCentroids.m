function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);

% Go over every example, find its closest centroid, and store
% the index inside idx at the appropriate location.
%               

for i = 1:length(X)
    Xcmp = [X(i,:)];
    for j = 2:K
        Xcmp = [Xcmp; X(i,:)];
    end
    sqrdif = ((centroids - Xcmp) * (centroids - Xcmp)').^(0.5);
    % only the diagonal components of sqrdif have the distances we need
    [dist ind] = min(diag(sqrdif));
    idx(i) = ind;
end


end

