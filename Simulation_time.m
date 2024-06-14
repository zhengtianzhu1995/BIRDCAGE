%% Simulation 1 consider different dimension p
% clean the workspace
clc
clear 
close 
warning off
rng('default')

% simulation setting
N = 50000;
p = 1000;
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
time1 = zeros(10,1);
time2 = zeros(10,1);
time3 = zeros(10,1);
time4 = zeros(10,1);

% replication
parfor (i = 1:10)
% data generation
[X, y, betaT, supp] = datagen2(N, p, opts);

% initial value
betainit = lasso(X,y);
betainit = betainit/norm(betainit)*norm(betaT);

% pool estimate
t1 = clock;
betaPOOL = pool(X, y, betainit, norm(betaT));
t2 = clock;
time1(i,1) = etime(t2,t1);

% median-of-means estimate
t3 = clock;
betaMOM = mom(X, y, betainit, norm(betaT));
t4 = clock;
time2(i,1) = etime(t4,t3);


% odist
t5 = clock;
betaOD = odist(X, y, betainit, norm(betaT));
t6 = clock;
time3(i,1) = etime(t6,t5);

% bdist
t7 = clock;
betaBD = bdist(X, y, betainit, norm(betaT));
t8 = clock;
time4(i,1) = etime(t8,t7);

end

% report
MEAN1 = median(time1);
MEAN2 = median(time2);
MEAN3 = median(time3);
MEAN4 = median(time4);


