%% Test

% clean the workspace
clc
clear 
close 
warning off
rng('default')
rng(4) % fix seed

% simulation setting
N = 10000;
n = 500;
L = N/n;

opts.c = 0;
opts.nt = 3;
opts.ct = 1;

K = 19;
tau = 0.05 * (1:K);

% test data generation
[X, y, betaT, supp] = datagen(N, opts);

% test initial value
betainit = lasso(X(1:n,:),y(1:n,:));
betainit = betainit/norm(betainit)*norm(betaT);

% test pool estimate
betaPOOL = pool(X, y, betainit, norm(betaT));
error1 = computeError(betaPOOL, betaT);
s1 = computeF1(betaPOOL, betaT);

% test median-of-means estimate
betaMOM = mom(X, y, betainit, norm(betaT));
error2 = computeError(betaMOM, betaT);
s2 = computeF1(betaMOM, betaT);

% test odist
betaOD = odist(X, y, betainit, norm(betaT));
error3 = computeError(betaOD, betaT);
s3 = computeF1(betaOD, betaT);

% test bdist
betaBD = bdist(X, y, betainit, norm(betaT));
error4 = computeError(betaBD, betaT);
s4 = computeF1(betaBD, betaT);





