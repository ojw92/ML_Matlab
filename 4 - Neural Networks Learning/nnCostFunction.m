function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params to be used as an input to our nnCostFunction and need to
%   be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% Theta1 takes the 1~(25*401) elements and converts to 25x401 matrix
% Theta2 makes 10x401 matrix with the rest of the elements
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, verify that the
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. Check that the implementation is correct
%         by running checkNNGradients
%
% Part 3: Implement regularization with the cost function and gradients.
%


% There are two ways to convert y into a 5000x10 permutation matrix
% that matches the values of y
% Method 1
% y_new = [y zeros(length(y),(num_labels - 1))];    
% for i = 1:length(y)
%     j = y_new(i,1)
%     y_new(i,:) = zeros(1, num_labels);
%     y_new(i,j) = 1;
% end   
    
% Method 2
y_mew = eye(size(X,1));
y_mew = y_mew(y,1:num_labels);

% To check both methods lend the same answer
% % If check == 50000 = 1, they're equal!
% check = y_new == y_mew;
% check == 50000;
    

X_bias = [ones(size(X,1), 1) X];
z2 = X_bias * Theta1';
a2 = sigmoid(X_bias * Theta1');
z3 = [ones(size(a2,1), 1) a2] * Theta2';
h = sigmoid([ones(size(a2,1), 1) a2] * Theta2');
% y_mew and h are both 5000x10, and
% we want to sum over all m & k, so we do element-wise multiplication
J = 1/m * sum(sum(-y_mew.*log(h) - (1-y_mew).*log(1-h)));

% With regularization (bias terms are removed from Thetas)
TNBias1 = Theta1(:, 2:end);
TNBias2 = Theta2(:, 2:end);
J = J + lambda/2/m * (sum(sum(TNBias1.*TNBias1)) + sum(sum(TNBias2.*TNBias2)));

% Back propagation to find "error caused by unit (which are columns)"
Error_h = h - y_mew;
Error_2 = Error_h * TNBias2 .* sigmoidGradient(z2); 

% Accumulate the gradients
% for t = 1:m
%     Theta2_grad = Theta2_grad + Error_h' * a2;
%     Theta1_grad = Theta1_grad + Error_2' * X_bias;
% end

% Do forward pass and backpropagation on one example at a time
for i = 1:m
    % Forward pass to get all the values
    z2(i,:) = X_bias(i,:) * Theta1';
    a2(i,:) = sigmoid(z2(i,:));
    z3(i,:) = [1 a2(i,:)] * Theta2';
    h(i,:) = sigmoid(z3(i,:));
    % Error for each example
    Error_h(i,:) = h(i,:) - y_mew(i,:);
    Error_2(i,:) = Error_h(i,:) * TNBias2 .* sigmoidGradient(z2(i,:));
    % Accumulate the gradients
    % Ignoring bias. T2grad=10x1 * 1x25; T1grad = 25*1 x 1*40;
    Theta2_grad = Theta2_grad + Error_h(i,:)' * [1 a2(i,:)];
    Theta1_grad = Theta1_grad + Error_2(i,:)' * [1 X(i,:)];
end

% With regularization terms at the end
% First columns of Thetas are not regularized (0's for bias to add 0 to grads)
Theta2_grad = 1/m * (Theta2_grad + lambda*([zeros(num_labels,1) TNBias2]));
Theta1_grad = 1/m * (Theta1_grad + lambda*([zeros(hidden_layer_size,1) TNBias1]));



% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
