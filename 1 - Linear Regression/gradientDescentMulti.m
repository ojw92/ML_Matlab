function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % Perform a single gradient step on the parameter vector theta
    %

%y_norm = (y - mean(y))/std(y);

%temp1 = theta(1) - alpha/m * (sum(X*theta - y)); 
%temp2 = theta(2) - alpha/m * sum(X(:,2)'*(X*theta - y));
%temp3 = theta(3) - alpha/m * sum(X(:,3)'*(X*theta - y)); 
%theta = [temp1; temp2; temp3]

tempo = zeros(length(theta),1);

for i = 1:length(theta),
    temp = theta(i) - alpha/m*sum(X(:,i)'*(X*theta - y));
    tempo(i) = temp;
end

theta = tempo;


    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);



end

end
