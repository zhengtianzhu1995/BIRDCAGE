function [beta] = pool_byz(X, y, init, nn, Bt)
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

[N,~] = size(X);

% all estimates
ithist.beta = [];
ithist.beta(:,1) = init;

% iteration
for i = 2:maxiter
    yt = psy(X, y, ithist.beta(:,(i-1)), tau, K);
    if Bt == 1
        yt = 10*randn(N,1);
    elseif Bt == 2
        yt = -100*yt;
    else
        X = -5*X;
    end
    beta = lasso(X,yt);
    beta = beta/norm(beta)*nn;
    ithist.beta(:,i) = beta;
end

end