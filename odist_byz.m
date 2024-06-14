function [beta] = odist_byz(X, y, init, nn, Bt, B)
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
L = N/n;

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
    Xbyz = X((1+(L-B)*n):N,:);
    yt = psy(X, y, ithist.beta(:,(ii-1)), tau, K);
    if Bt == 1
        yt((1+(L-B)*n):N,:) = 10*randn((B*n),1);
    elseif Bt == 2
        yt((1+(L-B)*n):N,:) = -100*yt((1+(L-B)*n):N,:);
    else
        Xbyz = -5*X((1+(L-B)*n):N,:);
    end
    zN = (X(1:(L-B)*n,:)'*yt(1:(L-B)*n,:) + Xbyz'*yt((1+(L-B)*n):N,:))/N;
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
