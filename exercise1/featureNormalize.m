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

mu = mean(X)
sigma = std(X)

mu = ones(size(X,1), 1)* mu;
sigma = ones(size(X,1),1)*sigma;
X_norm = (X-mu)./sigma;

%x1_mean = mean(X(:, 1));
%x1_std = std(X(:, 1));
%x2_mean = mean(X(:, 2));
%x2_std = std(X(:, 2));
%mu = [x1_mean x2_mean];
%mu = repmat(mu, size(X,1), 1);

%sigma = [x1_std x2_std];
%sigma = repmat(sigma, size(X,1), 1);
%X_norm = (X_norm-mu)./sigma

%X_norm_mean = [X_norm(:,1)-x1_mean X(:,2)-x2_mean];
%X_norm = [X_norm_mean(:,1)./x1_std X_norm_mean(:,2)./x2_std];
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









% ============================================================

end
