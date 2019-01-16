function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_partial = (1/(2*m))*(transpose(((X*theta)-y))*((X*theta)-y));
J_regularization = (lambda/(2*m)) * sum(theta(2:end).^2);
J = J_partial + J_regularization;


grad_regularization_1 = (1/m) * (X' * ((X*theta)-y));

grad_regularization = (lambda/m) .* theta(2:end);


grad_regularized = [0; grad_regularization];
grad =  grad_regularization_1 + grad_regularized;






% =========================================================================

grad = grad(:);

end
