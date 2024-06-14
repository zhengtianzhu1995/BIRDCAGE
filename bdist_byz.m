function [beta] = bdist_byz(X, y, init, nn, Bt, B)
% bdist Perform lasso regularization for linear regression with
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

Z = ones(p,L);

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
        for j = 1:(L-B)
            Z(:,j) = X((1+(j-1)*n):(j*n),:)'*yt((1+(j-1)*n):(j*n),:)/n;
        end
        for jj = (1+(L-B)):L
            Z(:,jj) = Xbyz((1+(jj-L+B-1)*n):(jj-L+B)*n,:)'*yt((1+(jj-1)*n):(jj*n),:)/n;
        end
    zN = median(Z,2);
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
