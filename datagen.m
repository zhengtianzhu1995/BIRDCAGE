function [X, y, betaT, supp] = datagen(N, opts, varargin)
%  datagen generates the N data scattered across L nodes in Example 1.
%  
%  Parameters:
%
%    N                A numeric scalar, the total sample size
%    L                A numeric scalar, the number of nodes
%
%  Optional input parameters:
%   'c'              The data contamination proportion
%   'nt'             The type of noise, 
%                    1 for normal, 2 for Cauchy, 3 for exponential,
%                    (update) 4 for t_2 distribution
%   'ct'             The data contamination type

%  Return values:
%    X                The covariables matrix which dimension is N*p.
%    y                The response vector of length N.
%    betaT            The true parameter of length p.
%    supp             The support set of betaT.

% fixed parameters
n = 500;
p = 500;
rho = 0.5;
L = N/n;

% check settings
if ~isfield(opts, 'c'); opts.c = 0; end
if ~isfield(opts, 'nt'); opts.nt = 1; end
if ~isfield(opts, 'ct'); opts.ct = 1; end
c = opts.c;
nt = opts.nt;
ct = opts.ct;

% --------------------------------------
% Generate data
% --------------------------------------

% true parameters
betaT = zeros(p,1); 
betaT(1,1) = 1;
betaT(4,1) = 1.5;
betaT(5,1) = 1.75;
betaT(7,1) = 2;
betaT(10,1) = 3;
supp = find(betaT);

% X
SIGMA = rho.^(abs(transpose(1:p)-(1:p)));
Mu = zeros(1,p);
X = mvnrnd(Mu,SIGMA,N);

% noise
if nt == 1
        noise = randn(N,1);
    elseif nt == 2
        noise = trnd(1,N,1);
    elseif nt == 3
        noise = exprnd(1,[N,1]);
    else
        noise = trnd(2,N,1);
end

% Y
y = X*betaT + noise;


% data contamination
idx = randperm(N,(c*n));

if c ~= 0
    if ct == 1
        X(idx,:) = 1;
        y(idx,:) = 1;
    elseif ct == 2
        X(idx,:) = 1;
        y(idx,:) = 10000;
    else
        X(idx,:) = rand((c*n),p);
        y(idx,:) = binornd(1,0.5,[(c*n),1]);
    end
else
    y = y;
end

end
