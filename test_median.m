p = 500;
rho = 0.5;
N = 10000;
n = 500;
L = 20;

% beta
betaT = zeros(p,1);
betaT(1:10,:) = 1;

% X
SIGMA = rho.^(abs(transpose(1:p)-(1:p)));
Mu = zeros(1,p);
X = mvnrnd(Mu,SIGMA,N);

% noise
noise = randn(N,1);

% Y
y = X*betaT + noise;

% mean 
MEAN = X'*y/N;

% median
M = ones(p,4);
M(:,1) = X(1:2500,:)'*y(1:2500,:)/2500;
M(:,2) = X(2501:5000,:)'*y(2501:5000,:)/2500;
M(:,3) = X(5001:7500,:)'*y(5001:7500,:)/2500;
M(:,4) = X(7501:10000,:)'*y(7501:10000,:)/2500;
M1 = mean(M,2);
M2 = median(M,2);


