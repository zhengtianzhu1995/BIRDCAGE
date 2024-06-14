%% Simulation 1 consider different dimension p
% clean the workspace
clc
clear 
close 
warning off
rng('default')

% simulation setting
N = 50000;
p = 800;
n = 500;
L = N/n;

% c = 0 for simulation 1
opts.c = 0;
opts.nt = 3;
opts.ct = 1;

% quantile
K = 19;
tau = 0.05 * (1:K);

% store the results
error1 = zeros(100,1);
error2 = zeros(100,1);
error3 = zeros(100,1);
error4 = zeros(100,1);
s1 = zeros(100,1);
s2 = zeros(100,1);
s3 = zeros(100,1);
s4 = zeros(100,1);

% replication
parfor (i = 1:100)
% data generation
[X, y, betaT, supp] = datagen2(N, p, opts);

% initial value
betainit = lasso(X,y);
betainit = betainit/norm(betainit)*norm(betaT);

% pool estimate
betaPOOL = pool(X, y, betainit, norm(betaT));
error1(i,1) = computeError(betaPOOL, betaT);
s1(i,1) = computeF1(betaPOOL, betaT);

% median-of-means estimate
betaMOM = mom(X, y, betainit, norm(betaT));
error2(i,1) = computeError(betaMOM, betaT);
s2(i,1) = computeF1(betaMOM, betaT);

% odist
betaOD = odist(X, y, betainit, norm(betaT));
error3(i,1) = computeError(betaOD, betaT);
s3(i,1) = computeF1(betaOD, betaT);

% bdist
betaBD = bdist(X, y, betainit, norm(betaT));
error4(i,1) = computeError(betaBD, betaT);
s4(i,1) = computeF1(betaBD, betaT);

end

% select
for j = 1:100
    if error4(j,1) > 0.5
        error4(j,1) = 0; 
        s4(j,1) = 0;
    end
end

% report
MEAN1 = mean(error1);
MEAN2 = mean(error2);
MEAN3 = mean(error3);
MEAN4 = sum(error4)/length(find(error4));

S1 = mean(s1);
S2 = mean(s2);
S3 = mean(s3);
S4 = sum(s4)/length(find(s4));

