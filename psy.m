function [psy] = psy(X, y, beta, tau, K)
% psy transform the original response to the pseudo response  
% Input:
%   X : original X
%   y : original y
%   beta : parameter to estimate
%   tau : a series of quantile

% get the sample size N
[N,~] = size(X);

% compute density 
[f,~] = ksdensity(y - X*beta, 0);

% compute pseudo y
E = repmat(y - X*beta, 1, K);
qt = (E <= 0) - repmat(tau, N, 1);

psy = X*beta - 1/f * mean(qt, 2);
end