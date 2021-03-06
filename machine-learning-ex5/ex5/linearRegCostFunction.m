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

% ====================== COST Calculation ======================
predictions = X*theta;
sqrErrors = (predictions-y).^2;
error=(X*theta)-y;

J_unreg=1/(2*m) * sum(sqrErrors);
%====================== Regularize J ====================== 
theta(1)=0;
J=J_unreg+(lambda/(2*m))*sum(theta.^2);


% ====================== Gradient Calculation ======================
% Instructions: Perform a single gradient step on the parameter vector
%               theta. 

grad_unreg=(1/m).*(X'*error);
grad=grad_unreg+((lambda/m).*theta);
% ============================================================
%size(J)
%grad








% =========================================================================

grad = grad(:);

end
