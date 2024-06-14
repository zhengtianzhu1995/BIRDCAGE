function [beta] = mom_byz(X, y, init, nn, Bt, B)
%   mom aggregates estimates by median-of-means
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

% get the size
[N,p] = size(X);
n = 500;
L = N/n;

% median-of-means
beta = ones(p,L);
for ii = 1:(L-B)
    beta(:,ii) = pool(X((1+(ii-1)*n):(ii*n),:), y((1+(ii-1)*n):(ii*n),:), init, nn);
    beta(:,ii) = force_first_positive(beta(:,ii));
end
for jj = (1+(L-B)):L
    beta(:,jj) = pool_byz(X((1+(ii-1)*n):(ii*n),:), y((1+(ii-1)*n):(ii*n),:), init, nn, Bt);
end
beta = median(beta,2);
beta = beta/norm(beta)*nn;

end