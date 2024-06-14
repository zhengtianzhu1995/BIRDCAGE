%% Simulation 3 add t_2 noises

% clean the workspace
clc
clear 
close 
warning off
rng('default')

% simulation setting
N = 20000;
n = 500;
L = N/n;
opts.c = 0;
opts.ct = 1;

% quantile
K = 19;
tau = 0.05 * (1:K);

% data generation
[X, y, betaT, supp] = datagen(N, opts);

% initial value
betainit = lasso(X,y);
betainit = betainit/norm(betainit)*norm(betaT);

% noise
opts.nt = 4;

% byz settings
B = 8;
Bt = 3;

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
[X, y, betaT, supp] = datagen(N, opts);

% initial value
betainit = lasso(X,y);
betainit = betainit/norm(betainit)*norm(betaT);

% pool estimate
betaPOOL = pool(X, y, betainit, norm(betaT));
error1(i,1) = computeError(betaPOOL, betaT);
s1(i,1) = computeF1(betaPOOL, betaT);

% median-of-means estimate
betaMOM_byz = mom_byz(X, y, betainit, norm(betaT), Bt, B);
error2(i,1) = computeError(betaMOM_byz, betaT);
s2(i,1) = computeF1(betaMOM_byz, betaT);

% odist
betaODbyz = odist_byz(X, y, betainit, norm(betaT), Bt, B);
error3(i,1) = computeError(betaODbyz, betaT);
s3(i,1) = computeF1(betaODbyz, betaT);

% bdist
betaBDbyz = bdist_byz(X, y, betainit, norm(betaT), Bt, B);
error4(i,1) = computeError(betaBDbyz, betaT);
s4(i,1) = computeF1(betaBDbyz, betaT);

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


