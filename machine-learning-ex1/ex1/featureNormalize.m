function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

mu = mean(X_norm);              % returns a row vector where each column is the mean of all the rows in that column of X
sigma = std(X_norm);            % returns a row vector where each column is the std dev of all the rows in that column of X
m = size(X_norm, 1);            % returns the number of rows in X
mu_matrix = ones(m, 1) * mu;  % m x 1 matrix multiplied with 1 x n(n:# of columns=3) returns a m x n matrix
sigma_matrix = ones(m, 1) * sigma; % m x 1 matrix multiplied with 1 x n(n:# of columns=3) returns a m x n matrix

X_norm = (X_norm-mu_matrix)./sigma_matrix; % returns a m x n normalized matrix without the 1st X0 (1's) column







% ============================================================

end
