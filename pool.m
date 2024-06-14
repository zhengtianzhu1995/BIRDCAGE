function [beta] = pool(X, y, init, nn)
%   pool solves the transformed L2-loss based on total observation
%  
%   Parameters:
%
%   X                A numeric matrix (dimension, say, Nxp)
%   y                A numeric vector of length N
%   init             A numeric vector length p, initial value 
%   nn               A scalar, the norm of true beta  
%
%   Return values:
%   beta             The estimate.

% default settings
K = 19;
tau = 0.05 * (1:K);
maxiter = 5;

% all estimates
ithist.beta = [];
ithist.beta(:,1) = init;

% iteration
for i = 2:maxiter
    yt = psy(X, y, ithist.beta(:,(i-1)), tau, K);
    beta = lasso(X,yt);
    beta = beta/norm(beta)*nn;
    ithist.beta(:,i) = beta;
end

end