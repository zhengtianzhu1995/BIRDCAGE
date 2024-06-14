function [beta] = odist(X, y, init, nn)
% odist Perform lasso regularization for linear regression with
%  data scattered across L nodes. The loss function is l2-loss.
%  
%  Parameters:
%
%    X                A numeric matrix (dimension, say, Nxp)
%    y                A numeric vector of length N
%    L                A numeric scalar, the number of nodes.
%
%  Optional input parameters:
%
%
%  Return values:
%    beta             The estimate.
%    ithist           The history of iterations.

K = 19;
tau = 0.05 * (1:K);
maxiter = 30;

tol = 0.05;

[N,p] = size(X);
n = 500;

% --------------------------------------
% Initials
% --------------------------------------

ithist.beta = [];
ithist.beta(:,1) = init;

bSig = X'*X/N;
bSig1 = X(1:n,:)'*X(1:n,:)/n;

% --------------------------------------
% Distributed estimate
% --------------------------------------

for ii = 2:maxiter
    yt = psy(X, y, ithist.beta(:,(ii-1)), tau, K);
    zN = X'*yt/N;
    b = zN + (bSig1-bSig)*ithist.beta(:,(ii-1));
    beta = pdasc(bSig1,b,n,p);
    beta = beta/norm(beta)*nn;
    if norm(beta - ithist.beta(:,(ii-1))) < tol
        break
    else 
        ithist.beta(:,ii) = beta;
    end
end
end %-lassoDistributed
