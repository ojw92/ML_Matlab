function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

Z = zeros(size(X, 1), K);

% Compute the projection of the data using only the top K 
% eigenvectors in U (first K columns). 
%               

for i = 1:size(X,1)
    for k = 1:K
        Z(i,k) = X(i, :) * U(:, k);
    end
end


end
