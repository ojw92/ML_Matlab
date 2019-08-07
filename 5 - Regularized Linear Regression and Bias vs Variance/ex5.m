%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance


%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  Start by first loading and visualizing the dataset. 
%  Load the dataset into your environment and plot the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% There will be X, y, Xval, yval, Xtest, ytest in the environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  Implement the cost function for regularized linear regression 
%

theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============
%  Implement the gradient for regularized linear regression
%

theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Train regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
%

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Implement the learningCurve function.
%
%  Write Up Note: Since the model is underfitting the data, expect to
%                 see a graph with "high bias" 
%

lambda = 0;
[error_train, error_val] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. 
%  Map each example into its powers
%

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Experiment with polynomial regression with multiple values of lambda
$  Run polynomial regression with lambda = 0
%

lambda = 0;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(1);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(2);
[error_train, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:m, error_train, 1:m, error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Train', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:m
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  Implement validationCurve to test various values of 
%  lambda on a validation set. Then select the "best" lambda value
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;


% Optional (ungraded) exercise: Computing test set error
% Computing the test error using the lambda found
Theta = trainLinearReg(X, y, lambda_vec(error_val == min(error_val)));
error_test = linearRegCostFunction(Xtest, ytest, Theta, 0);
fprintf('Best lambda was %f and its corresponding error_val was %f \n', ...
    lambda_vec(error_val == min(error_val)), min(error_val));
fprintf('Using this lambda, our test error was %f \n', error_test);


% Optional (ungraded) exercise: Plotting learning curves with randomly selected examples
% In practice, especially for small training sets, when you plot learning curves to
% debug your algorithms, it is often helpful to average across multiple sets of randomly
% selected examples to determine the training error and cross validation error.
% Each column of the following matrices will have 1 iteration of learning curve data
% We will then average all iterations, and get 2 vectors that we'll plot
error_train_X = zeros(length(X)*2/3, 1);
error_val_Xval = zeros(length(X)*2/3, 1);      % both matrices must have same # of rows
lambda_a = 0.01;
num_iter = 10;          % number of iterations should be 50; use 10 for quick checking

for i = 1:num_iter
    selX = randperm(size(X, 1));         % random indices with length X
    X_r = X(selX(1:(length(X)*2/3)));      % pick 2/3 of 12 datapoints of X based on these indices
    y_r = y(selX(1:(length(y)*2/3)));      % picking corresponding y for these X elements
    X_rp = polyFeatures(X_r, 8);         % creating polynomial features for our examples
    X_rp = featureNormalize(X_rp); % No need for [X_norm,mu,sigma] on left side, if we just want X_norm
    selXval = randperm(size(Xval,1));    % also picking examples from cross validation set
    Xval_r = Xval(selXval(1:(length(Xval)*2/3)));
    yval_r = yval(selXval(1:(length(yval)*2/3)));
    Xval_rp = polyFeatures(Xval_r, 8);
    Xval_rp = featureNormalize(Xval_rp);
    
    % If you want to do linear fit, calculate theta using trainLinearReg, then
    % run the plotFit function. But here, we just want the learning curves.
    %[error_train_X(:,i), error_val_Xval(:,i)] = learningCurve(X_rp, y_r, Xval_rp, yval_r, lambda_a);
    % Must add the bias terms before training theta!
    [err_train, err_val] = learningCurve([ones(size(X_rp,1),1) X_rp], y_r, [ones(size(Xval_rp,1),1) Xval_rp], yval_r, lambda_a);
    error_train_X = error_train_X + err_train;
    error_val_Xval = error_val_Xval + err_val;
end

% mean(A,2) gives column vector where each element is average of elements of that row
% mean(A) = mean(A,1) = row vector where each column is average of that column
% bsxfun(@rdivide, A, num_iter) basically divides by sample size (average), or A/num_iter
error_train_X = bsxfun(@rdivide, error_train_X, num_iter);  
error_val_Xval = bsxfun(@rdivide, error_val_Xval, num_iter);

figure(2);
plot(1:m*2/3, error_train_X, 1:m*2/3, error_val_Xval);
title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda_a));
xlabel('Number of training examples')
ylabel('Error')
axis([0 10 0 100])
legend('Train', 'Cross Validation')
