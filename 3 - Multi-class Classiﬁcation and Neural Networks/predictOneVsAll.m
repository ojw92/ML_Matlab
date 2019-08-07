function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Make predictions using the learned logistic regression parameters (one-vs-all)

% all_theta stores the trained theta values from 100 randomly selected examples
% X*all_theta' makes predictions on 5000 examples by giving 10 probabilities in each row,
% each value for the likeliness to belong to that classification, for all 5000 examples
% z is thus 5000 x 10, and p = indices of the max of each row
% we can write the code so that the max function returns 2 values:
% z for the max values, and p for the indices of those values
z = sigmoid(X*all_theta');
[z p] = max(z, [], 2);


% f = find(z == max(z, [], 2))
% for i = 1:5000
%     if f(i) <= 5000
%         p(f(i)) = 1;
%     elseif 5001 <= f(i) <= 10000
%         p(f(i)-5000) = 2;
%     elseif 10001 <= f(i) <= 15000
%         p(f(i)-10000) = 3;
%     elseif 15001 <= f(i) <= 20000
%         p(f(i)-15000) = 4;
%     elseif 20001 <= f(i) <= 25000
%         p(f(i)-20000) = 5;
%     elseif 25001 <= f(i) <= 30000
%         p(f(i)-25000) = 6;
%     elseif 30001 <= f(i) <= 35000
%         p(f(i)-30000) = 7;
%     elseif 35001 <= f(i) <= 40000
%         p(f(i)-35000) = 8;
%     elseif 40001 <= f(i) <= 45000
%         p(f(i)-40000) = 9;
%     elseif 45001 <= f(i) <= 50000
%         p(f(i)-45000) = 0;
%     end
% end


% ...
% for f <= 5000 && i = 1:5000
%     if f(i) <= 5000
%         p(i) = 1
%     end
% end


end
