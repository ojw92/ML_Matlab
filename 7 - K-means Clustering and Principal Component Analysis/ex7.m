%% Machine Learning Online Class
%  Exercise 7 | Principle Component Analysis and K-Means Clustering
%
%

%% Initialization
clear ; close all; clc

%% ================= Part 1: Find Closest Centroids ====================
%  Find closest centroids
%
fprintf('Finding closest centroids.\n\n');

% Load an example dataset that we will be using
load('ex7data2.mat');

% Select an initial set of centroids
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];

% Find the closest centroids for the examples using the
% initial_centroids
idx = findClosestCentroids(X, initial_centroids);

fprintf('Closest centroids for the first 3 examples: \n')
fprintf(' %d', idx(1:3));
fprintf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ===================== Part 2: Compute Means =========================
%  After implementing the closest centroids function,
%  complete the computeCentroids function.
%
fprintf('\nComputing centroids means.\n\n');

%  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K);

fprintf('Centroids computed after initial finding of closest centroids: \n')
fprintf(' %f %f \n' , centroids');
fprintf('\n(the centroids should be\n');
fprintf('   [ 2.428301 3.157924 ]\n');
fprintf('   [ 5.813503 2.633656 ]\n');
fprintf('   [ 7.119387 3.616684 ]\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 3: K-Means Clustering ======================
%  Run the K-Means algorithm 
%

fprintf('\nRunning K-Means clustering on example dataset.\n\n');

% Load an example dataset
load('ex7data2.mat');

% Settings for running K-Means
K = 3;
max_iters = 10;

% For consistency, here centroids are set to specific values
% but in practice it is desirable to generate them automatically, 
% such as by settings them to be random examples (as can be seen in
% kMeansInitCentroids).
initial_centroids = [3 3; 6 2; 8 5];

% Run K-Means algorithm. The 'true' at the end tells the function to plot
% the progress of K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters, true);
fprintf('\nK-Means Done.\n\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: K-Means Clustering on Pixels ===============
%  Use K-Means to compress an image

fprintf('\nRunning K-Means clustering on pixels from an image.\n\n');

%  Load an image of a bird
A = double(imread('bird_small.png'));

% If imread does not work, then try:
%   load ('bird_small.mat');

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
% This gives us our dataset matrix X that we will use K-Means on.
X = reshape(A, img_size(1) * img_size(2), 3);

% Run the K-Means algorithm on this data
% Try different values of K and max_iters here
K = 16; 
max_iters = 10;

% When using K-Means, it is important the initialize the centroids
% randomly. 
% Complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================= Part 5: Image Compression ======================
%  Use the clusters of K-Means to ompress an image


fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(X, centroids);

% Essentially, image X has been represented as in terms of the
% indices in idx. 

% Recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
X_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));


fprintf('Program paused. Press enter to continue.\n');
pause;


hold on;
figure(2);


% Trying image compression on my own image
fprintf('\nRunning K-Means clustering on pixels from an image of my own.\n\n');

JoJo = double(imread("GiveAShot.PNG"));

JoJo = JoJo / 255; % Divide by 255 so that pixel intensity values are in the range 0 - 1
J_img_size = size(JoJo); % size of the image

% Reshape the image into an Nx3 matrix where N = number of pixels.
% Each row will contain the Red, Green and Blue pixel values
J = reshape(JoJo, J_img_size(1) * J_img_size(2), 3);

K = 20; 
max_iters = 15;

% When using K-Means, it's important the initialize the centroids randomly.
initial_centroids = kMeansInitCentroids(J, K);

% Run K-Means
[centroids, idx] = runkMeans(J, initial_centroids, max_iters);

fprintf('Program paused. Press enter to continue.\n');
pause;





fprintf('\nApplying K-Means to compress an image.\n\n');

% Find closest cluster members
idx = findClosestCentroids(J, centroids);

% Now JoJo's image J is represented in terms of the indices in idx.

% We can now recover the image from the indices (idx) by mapping each pixel
% (specified by its index in idx) to the centroid value
J_recovered = centroids(idx,:);

% Reshape the recovered image into proper dimensions
J_recovered = reshape(J_recovered, J_img_size(1), J_img_size(2), 3);

% Display the original JoJo 
subplot(1, 2, 1);
imagesc(JoJo); 
title('Original');

% Display compressed JoJo side-by-side
subplot(1, 2, 2);
imagesc(J_recovered)
title(sprintf('Compressed, with %d colors.', K));


fprintf('Program paused. Press enter to continue.\n');
pause;
