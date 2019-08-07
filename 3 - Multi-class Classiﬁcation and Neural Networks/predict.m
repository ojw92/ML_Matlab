function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
p = zeros(size(X, 1), 1);

% Complete the following code to make predictions using the
% learned neural network. 


X = [ones(m, 1) X];
a2 = [ones(m, 1) sigmoid(X*Theta1')];
a3 = sigmoid(a2*Theta2');
[h p] = max(a3, [], 2)
% Adding ones for bias unit before each calculation
% h is the max of probability value, p is the index (classification) on row


end
