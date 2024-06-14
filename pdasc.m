function [beta, lam, ithist] = pdasc(A, b, n, p, opts)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %             min      1/2*beta^T*A*beta-beta^T*b  + lambda ||beta||_1  %
    %    by PDAS  algorithm with  continuation Lam ={lam_{1},...,lam_{N}    %
    %                              
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % INPUTS:                                                               %
    %            A  ---  Input matrix (R^{p*p})                             %
    %            b  ---  Input vector                                       %
    %        opts ---  structure containing                                 %
    %         N1   --- length of path (default: 200)                        %
    %        mu   --- stop if \|beta_{lam_k}\|_0 > mu                       %
    %       init  --- initial value for beta (default: 0)                   %
    % OUTPUTS:                                                              %
    %     beta   ---- recovered signal                                      %
    %    lam  ---- regularization  parameter                                %
    %  ithist ---- structure on iteration history, containing               %
    %          .beta  --- solution path                                     %
    %          .as --- size of active set  on the path                      %
    %          .it --- # of iteration on the path                           %
    % ======================================================================%

    linf = norm(b, inf);

   
        opts.N1 = 300;
        opts.mu = n / log(p);
        opts.Lmax = 1;
        opts.Lmin = 1e-4;
        opts.init = zeros(p, 1);
        opts.p = p;
        opts.n = n;
    

    % construct the homotopy path
    Lam = exp(linspace(log(opts.Lmax), log(opts.Lmin), opts.N1))';
    Lam = Lam(2:end);
    Lam = Lam * linf;
    ithist.Lam = Lam;
    %% main loop for pathfolling and choosing lambda and output solution
    ithist.beta = [];
    ithist.as = [];

    for k = 1:length(Lam)
        opts.lam = Lam(k);
        [beta, s] = pdas(A, b, opts);
        opts.init = beta;
        ithist.beta(:, k) = beta;
        ithist.as = [ithist.as; s]; % size of active set

        if s > opts.mu
            % display('# NON-ZERO IS TOO MUCH, STOP ...')
            break
        end

    end

    % select the solution on the path by voting
    ii = find(ithist.as == mode(ithist.as));
    ii = ii(end);
    beta = ithist.beta(:, ii);
    lam = Lam(ii);
end %-pdasc

function [beta, s] = pdas(A, b, opts)
    %-------------------------------------------------------------------------%
    %         Solving                                                         %
    %           1/2*beta^T*A*beta-beta^T*b  + lambda ||beta||_1               %
    %         by  one step primal-dual active set algorithm                   %
    %-------------------------------------------------------------------------%
    % INPUTS:                                                                 %
    %            A  ---  Input matrix (R^{p*p})                               %
    %          opts ---  structure containing                                 %
    %          lam  ---  regualrization paramter                              %
    %         init  ---  initial guess                                        %
    % OUTPUTS:                                                                %
    %         beta  ---  solution                                             %                                                                         %
    %         s     ---  size of active set                                   %
    %-------------------------------------------------------------------------%
    lam = opts.lam;
    beta0 = opts.init;
    p = opts.p;
    % initializing ...
    pd = beta0 + (b - A * beta0); % initial guesss of d
    Ac = find(abs(pd) > lam); % active set
    s = length(Ac);
    beta = zeros(p, 1);
    dAc = lam * sign(pd(Ac));
    bAc = b(Ac);
    rhs = bAc - dAc;
    G = A(Ac, Ac);
    beta(Ac) = G \ rhs;
end %-pdas



