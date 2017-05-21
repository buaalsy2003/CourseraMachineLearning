function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

tmp = sigmoid(X*theta);

theta_zeroed_first = [0; theta(2:length(theta));];

J = (-y' * log(tmp) .- (1 .- y')*log(1-tmp))/m+lambda/(2*m)*sum(theta_zeroed_first.^2);
grad = X'*(tmp - y)/m .+ (lambda / m) * theta_zeroed_first;




% =============================================================

end
