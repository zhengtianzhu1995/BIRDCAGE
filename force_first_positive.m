function beta1 = force_first_positive(beta)
% Force the first non-zero element of beta
% to be positive
% ------------------------------------------------------ %
    idx = find(beta); % indices of non-zero elements of betaT
    if ~isempty(beta)
        beta1 = beta*beta(idx(1)); % enforce the first non-zero element is positive
    else
        beta1 = beta1;
    end
end

